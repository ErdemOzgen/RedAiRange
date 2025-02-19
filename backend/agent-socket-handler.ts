import { RedAiRangeServer } from "./redairange-server";
import { AgentSocket } from "../common/agent-socket";
import { RedAiRangeSocket } from "./util-server";

export abstract class AgentSocketHandler {
    abstract create(socket : RedAiRangeSocket, server : RedAiRangeServer, agentSocket : AgentSocket): void;
}
