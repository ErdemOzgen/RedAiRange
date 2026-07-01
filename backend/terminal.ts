import { RedAiRangeServer } from "./redairange-server";
import * as os from "node:os";
import * as pty from "@homebridge/node-pty-prebuilt-multiarch";
import { LimitQueue } from "./utils/limit-queue";
import { RedAiRangeSocket } from "./util-server";
import {
    allowedCommandList, allowedRawKeys,
    PROGRESS_TERMINAL_ROWS,
    TERMINAL_COLS,
    TERMINAL_ROWS
} from "../common/util-common";
import { sync as commandExistsSync } from "command-exists";
import { log } from "./log";

const BACKSPACE_KEY = "\u007f";
const CTRL_C_KEY = "\u0003";
const ENTER_KEY = "\u000d";
const DEFAULT_TERM_TYPE = process.env.REDAIRANGE_TERM || "xterm-256color";

function getAllowedConsoleCommandSet() : Set<string> {
    const configured = process.env.REDAIRANGE_CONSOLE_ALLOWLIST;

    if (!configured || configured.trim().length === 0) {
        return new Set(allowedCommandList.map((command) => command.toLowerCase()));
    }

    return new Set(
        configured
            .split(",")
            .map((command) => command.trim().toLowerCase())
            .filter(Boolean)
    );
}

/**
 * Terminal for running commands, no user interaction
 */
export class Terminal {

    protected static terminalMap : Map<string, Terminal> = new Map();

    protected _ptyProcess? : pty.IPty;
    protected server : RedAiRangeServer;
    protected buffer : LimitQueue<string> = new LimitQueue(100);
    protected _name : string;

    protected file : string;
    protected args : string | string[];
    protected cwd : string;
    protected callback? : (exitCode : number) => void;

    protected _rows : number = TERMINAL_ROWS;
    protected _cols : number = TERMINAL_COLS;

    public enableKeepAlive : boolean = false;
    protected keepAliveInterval? : NodeJS.Timeout;
    protected kickDisconnectedClientsInterval? : NodeJS.Timeout;

    protected socketList : Record<string, RedAiRangeSocket> = {};

    constructor(server : RedAiRangeServer, name : string, file : string, args : string | string[], cwd : string) {
        this.server = server;
        this._name = name;
        //this._name = "terminal-" + Date.now() + "-" + getCryptoRandomInt(0, 1000000);
        this.file = file;
        this.args = args;
        this.cwd = cwd;

        Terminal.terminalMap.set(this.name, this);
    }

    get rows() {
        return this._rows;
    }

    set rows(rows : number) {
        this._rows = rows;
        try {
            this.ptyProcess?.resize(this.cols, this.rows);
        } catch (e) {
            if (e instanceof Error) {
                log.debug("Terminal", "Failed to resize terminal: " + e.message);
            }
        }
    }

    get cols() {
        return this._cols;
    }

    set cols(cols : number) {
        this._cols = cols;
        log.debug("Terminal", `Terminal cols: ${this._cols}`); // Added to check if cols is being set when changing terminal size.
        try {
            this.ptyProcess?.resize(this.cols, this.rows);
        } catch (e) {
            if (e instanceof Error) {
                log.debug("Terminal", "Failed to resize terminal: " + e.message);
            }
        }
    }

    public start() {
        if (this._ptyProcess) {
            return;
        }

        this.kickDisconnectedClientsInterval = setInterval(() => {
            for (const socketID in this.socketList) {
                const socket = this.socketList[socketID];
                if (!socket.connected) {
                    log.debug("Terminal", "Kicking disconnected client " + socket.id + " from terminal " + this.name);
                    this.leave(socket);
                }
            }
        }, 60 * 1000);

        if (this.enableKeepAlive) {
            log.debug("Terminal", "Keep alive enabled for terminal " + this.name);

            // Close if there is no clients
            this.keepAliveInterval = setInterval(() => {
                const numClients = Object.keys(this.socketList).length;

                if (numClients === 0) {
                    log.debug("Terminal", "Terminal " + this.name + " has no client, closing...");
                    this.close();
                } else {
                    log.debug("Terminal", "Terminal " + this.name + " has " + numClients + " client(s)");
                }
            }, 60 * 1000);
        } else {
            log.debug("Terminal", "Keep alive disabled for terminal " + this.name);
        }

        try {
            const spawnEnv = {
                ...process.env,
            };

            if (this.file === "docker") {
                const args = Array.isArray(this.args) ? this.args : [ this.args ];
                if (args.includes("compose") && !spawnEnv.COMPOSE_ANSI) {
                    spawnEnv.COMPOSE_ANSI = process.env.REDAIRANGE_COMPOSE_ANSI || "never";
                }
            }

            this._ptyProcess = pty.spawn(this.file, this.args, {
                // PTY "name" maps to TERM. Use a real terminal type, not the dynamic session name.
                name: DEFAULT_TERM_TYPE,
                cwd: this.cwd,
                cols: TERMINAL_COLS,
                rows: this.rows,
                env: spawnEnv,
            });

            // On Data
            this._ptyProcess.onData((data) => {
                this.buffer.pushItem(data);

                for (const socketID in this.socketList) {
                    const socket = this.socketList[socketID];
                    socket.emitAgent("terminalWrite", this.name, data);
                }
            });

            // On Exit
            this._ptyProcess.onExit(this.exit);
        } catch (error) {
            if (error instanceof Error) {
                clearInterval(this.keepAliveInterval);

                log.error("Terminal", "Failed to start terminal: " + error.message);
                const exitCode = Number(error.message.split(" ").pop());
                this.exit({
                    exitCode,
                });
            }
        }
    }

    /**
     * Exit event handler
     * @param res
     */
    protected exit = (res : {exitCode: number, signal?: number | undefined}) => {
        for (const socketID in this.socketList) {
            const socket = this.socketList[socketID];
            socket.emitAgent("terminalExit", this.name, res.exitCode);
        }

        // Remove all clients
        this.socketList = {};

        Terminal.terminalMap.delete(this.name);
        log.debug("Terminal", "Terminal " + this.name + " exited with code " + res.exitCode);

        clearInterval(this.keepAliveInterval);
        clearInterval(this.kickDisconnectedClientsInterval);

        if (this.callback) {
            this.callback(res.exitCode);
        }
    };

    public onExit(callback : (exitCode : number) => void) {
        this.callback = callback;
    }

    public join(socket : RedAiRangeSocket) {
        this.socketList[socket.id] = socket;
    }

    public leave(socket : RedAiRangeSocket) {
        delete this.socketList[socket.id];
    }

    public get ptyProcess() {
        return this._ptyProcess;
    }

    public get name() {
        return this._name;
    }

    /**
     * Get the terminal output string
     */
    getBuffer() : string {
        if (this.buffer.length === 0) {
            return "";
        }
        return this.buffer.join("");
    }

    close() {
        clearInterval(this.keepAliveInterval);
        // Send Ctrl+C to the terminal
        this.ptyProcess?.write("\x03");
    }

    /**
     * Get a running and non-exited terminal
     * @param name
     */
    public static getTerminal(name : string) : Terminal | undefined {
        return Terminal.terminalMap.get(name);
    }

    public static getOrCreateTerminal(server : RedAiRangeServer, name : string, file : string, args : string | string[], cwd : string) : Terminal {
        // Since exited terminal will be removed from the map, it is safe to get the terminal from the map
        let terminal = Terminal.getTerminal(name);
        if (!terminal) {
            terminal = new Terminal(server, name, file, args, cwd);
        }
        return terminal;
    }

    public static exec(server : RedAiRangeServer, socket : RedAiRangeSocket | undefined, terminalName : string, file : string, args : string | string[], cwd : string) : Promise<number> {
        return new Promise((resolve, reject) => {
            // check if terminal exists
            if (Terminal.terminalMap.has(terminalName)) {
                reject("Another operation is already running, please try again later.");
                return;
            }

            let terminal = new Terminal(server, terminalName, file, args, cwd);
            terminal.rows = PROGRESS_TERMINAL_ROWS;

            if (socket) {
                terminal.join(socket);
            }

            terminal.onExit((exitCode : number) => {
                resolve(exitCode);
            });
            terminal.start();
        });
    }

    public static getTerminalCount() {
        return Terminal.terminalMap.size;
    }
}

/**
 * Interactive terminal
 * Mainly used for container exec
 */
export class InteractiveTerminal extends Terminal {
    public write(input : string) {
        this.ptyProcess?.write(input);
    }

    resetCWD() {
        const cwd = process.cwd();
        this.ptyProcess?.write(`cd "${cwd}"\r`);
    }
}

/**
 * User interactive terminal that use bash or powershell with limited commands such as docker, ls, cd, dir
 */
export class MainTerminal extends InteractiveTerminal {
    private currentCommandInput = "";
    private commandInputDirty = false;

    constructor(server : RedAiRangeServer, name : string) {
        let shell;

        if (os.platform() === "win32") {
            if (commandExistsSync("pwsh.exe")) {
                shell = "pwsh.exe";
            } else {
                shell = "powershell.exe";
            }
        } else {
            shell = "bash";
        }
        super(server, name, shell, [], server.stacksDir);
    }

    public write(input : string) {
        if (input === CTRL_C_KEY) {
            this.currentCommandInput = "";
            this.commandInputDirty = false;
            super.write(input);
            return;
        }

        if (input === BACKSPACE_KEY) {
            this.currentCommandInput = this.currentCommandInput.slice(0, -1);
            super.write(input);
            return;
        }

        if (input.includes(ENTER_KEY)) {
            if (this.commandInputDirty) {
                this.currentCommandInput = "";
                this.commandInputDirty = false;
                throw new Error("Command editing shortcuts are disabled in restricted console.");
            }

            const inputParts = input.split(ENTER_KEY);
            this.currentCommandInput += inputParts[0] || "";

            const executable = this.currentCommandInput.trim().split(/\s+/)[0]?.toLowerCase();
            this.currentCommandInput = "";

            if (executable) {
                const allowedCommandSet = getAllowedConsoleCommandSet();

                if (!allowedCommandSet.has(executable)) {
                    log.warn("console", `Blocked command: ${executable}`);
                    throw new Error(`Command is not allowed: ${executable}`);
                }
            }

            super.write(input);
            return;
        }

        // For like Ctrl + C
        if (allowedRawKeys.includes(input)) {
            this.commandInputDirty = true;
            super.write(input);
            return;
        }

        if (this.currentCommandInput.length === 0) {
            this.commandInputDirty = false;
        }
        this.currentCommandInput += input;
        super.write(input);
    }
}
