import { RedAiRangeServer } from "./redairange-server";
import { log } from "./log";

log.info("server", "Welcome to redairange!");
const server = new RedAiRangeServer();
await server.serve();
