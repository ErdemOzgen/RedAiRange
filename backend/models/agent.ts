import { BeanModel } from "redbean-node/dist/bean-model";
import { R } from "redbean-node";
import { LooseObject } from "../../common/util-common";
import {
    decryptAgentCredential,
    encryptAgentCredential,
    isEncryptedAgentCredential,
} from "../agent-credential";

export class Agent extends BeanModel {

    static async getAgentList() : Promise<Record<string, Agent>> {
        let list = await R.findAll("agent") as Agent[];
        let result : Record<string, Agent> = {};
        for (let agent of list) {
            result[agent.endpoint] = agent;
        }
        return result;
    }

    get endpoint() : string {
        let obj = new URL(this.url);
        return obj.host;
    }

    static async encryptPassword(password: string): Promise<string> {
        return encryptAgentCredential(password);
    }

    static async decryptPassword(password: string): Promise<string> {
        return decryptAgentCredential(password);
    }

    static async getConnectionPassword(agent: Agent): Promise<string> {
        const plainPassword = await Agent.decryptPassword(agent.password);

        if (!isEncryptedAgentCredential(agent.password)) {
            const encrypted = await Agent.encryptPassword(agent.password);
            await R.exec("UPDATE `agent` SET password = ? WHERE id = ? ", [
                encrypted,
                agent.id,
            ]);
            agent.password = encrypted;
        }

        return plainPassword;
    }

    toJSON() : LooseObject {
        return {
            url: this.url,
            username: this.username,
            endpoint: this.endpoint,
        };
    }

}

export default Agent;
