import { RedAiRangeServer } from "./redairange-server";
import { RedAiRangeSocket } from "./util-server";

export abstract class SocketHandler {
    abstract create(socket : RedAiRangeSocket, server : RedAiRangeServer): void;
}
