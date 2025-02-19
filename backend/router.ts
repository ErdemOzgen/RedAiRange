import { RedAiRangeServer } from "./redairange-server";
import { Express, Router as ExpressRouter } from "express";

export abstract class Router {
    abstract create(app : Express, server : RedAiRangeServer): ExpressRouter;
}
