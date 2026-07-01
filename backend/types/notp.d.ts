declare module "notp" {
    interface TotpVerifyOptions {
        window?: number;
        time?: number;
    }

    interface TotpVerifyResult {
        delta: number;
    }

    interface Notp {
        totp: {
            verify(token: string, key: string, options?: TotpVerifyOptions): TotpVerifyResult | null;
        };
    }

    const notp: Notp;
    export default notp;
}
