use super::OpsMode;

#[derive(PartialEq, PartialOrd, Clone, Debug)]
pub struct SatRec {
    pub epochyr: i32,
    pub epochtynumrev: i32,
    pub error: i32,
    pub operationmode: OpsMode,
    pub init: char,
    pub method: char,

    /* Near Earth */
    pub isimp: i32,
    pub aycof: f64,
    pub con41: f64,
    pub cc1: f64,
    pub cc4: f64,
    pub cc5: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
    pub delmo: f64,
    pub eta: f64,
    pub argpdot: f64,
    pub omgcof: f64,
    pub sinmao: f64,
    pub t: f64,
    pub t2cof: f64,
    pub t3cof: f64,
    pub t4cof: f64,
    pub t5cof: f64,
    pub x1mth2: f64,
    pub x7thm1: f64,
    pub mdot: f64,
    pub nodedot: f64,
    pub xlcof: f64,
    pub xmcof: f64,
    pub nodecf: f64,

    /* Deep Space */
    pub irez: i32,
    pub d2201: f64,
    pub d2211: f64,
    pub d3210: f64,
    pub d3222: f64,
    pub d4410: f64,
    pub d4422: f64,
    pub d5220: f64,
    pub d5232: f64,
    pub d5421: f64,
    pub d5433: f64,
    pub dedt: f64,
    pub del1: f64,
    pub del2: f64,
    pub del3: f64,
    pub didt: f64,
    pub dmdt: f64,
    pub dnodt: f64,
    pub domdt: f64,
    pub e3: f64,
    pub ee2: f64,
    pub peo: f64,
    pub pgho: f64,
    pub pho: f64,
    pub pinco: f64,
    pub plo: f64,
    pub se2: f64,
    pub se3: f64,
    pub sgh2: f64,
    pub sgh3: f64,
    pub sgh4: f64,
    pub sh2: f64,
    pub sh3: f64,
    pub si2: f64,
    pub si3: f64,
    pub sl2: f64,
    pub sl3: f64,
    pub sl4: f64,
    pub gsto: f64,
    pub xfact: f64,
    pub xgh2: f64,
    pub xgh3: f64,
    pub xgh4: f64,
    pub xh2: f64,
    pub xh3: f64,
    pub xi2: f64,
    pub xi3: f64,
    pub xl2: f64,
    pub xl3: f64,
    pub xl4: f64,
    pub xlamo: f64,
    pub zmol: f64,
    pub zmos: f64,
    pub atime: f64,
    pub xli: f64,
    pub xni: f64,

    /*Elements */
    pub a: f64,
    pub altp: f64,
    pub alta: f64,
    pub epochdays: f64,
    pub jdsatepoch: f64,
    pub jdsatepoch_f: f64,
    pub nddot: f64,
    pub ndot: f64,
    pub bstar: f64,
    pub rcse: f64,
    pub inclo: f64,
    pub nodeo: f64,
    pub ecco: f64,
    pub argpo: f64,
    pub mo: f64,
    pub no: f64,
    pub no_kozai: f64,
    pub no_unkozai: f64,

    /* Gravity */
    pub tumin: f64,
    pub mus: f64,
    pub radiusearthkm: f64,
    pub xke: f64,
    pub j2: f64,
    pub j3: f64,
    pub j4: f64,
    pub j3oj2: f64,

    /* Mean elements */
    pub am: f64,
    pub em: f64,
    pub im: f64,
    pub om: f64,
    pub mm: f64,
    pub nm: f64,
}

impl Default for SatRec {
    fn default() -> Self {
        Self::new()
    }
}

impl SatRec {
    pub const fn new() -> Self {
        Self {
            epochyr: 0,
            epochtynumrev: 0,
            error: 0,
            operationmode: OpsMode::IMPROVED,
            init: 'n',
            method: 'm',

            isimp: 0,
            aycof: 0.0,
            con41: 0.0,
            cc1: 0.0,
            cc4: 0.0,
            cc5: 0.0,
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
            delmo: 0.0,
            eta: 0.0,
            argpdot: 0.0,
            omgcof: 0.0,
            sinmao: 0.0,
            t: 0.0,
            t2cof: 0.0,
            t3cof: 0.0,
            t4cof: 0.0,
            t5cof: 0.0,
            x1mth2: 0.0,
            x7thm1: 0.0,
            mdot: 0.0,
            nodedot: 0.0,
            xlcof: 0.0,
            xmcof: 0.0,
            nodecf: 0.0,
            /* Deep Space */
            irez: 0,
            d2201: 0.0,
            d2211: 0.0,
            d3210: 0.0,
            d3222: 0.0,
            d4410: 0.0,
            d4422: 0.0,
            d5220: 0.0,
            d5232: 0.0,
            d5421: 0.0,
            d5433: 0.0,
            dedt: 0.0,
            del1: 0.0,
            del2: 0.0,
            del3: 0.0,
            didt: 0.0,
            dmdt: 0.0,
            dnodt: 0.0,
            domdt: 0.0,
            e3: 0.0,
            ee2: 0.0,
            peo: 0.0,
            pgho: 0.0,
            pho: 0.0,
            pinco: 0.0,
            plo: 0.0,
            se2: 0.0,
            se3: 0.0,
            sgh2: 0.0,
            sgh3: 0.0,
            sgh4: 0.0,
            sh2: 0.0,
            sh3: 0.0,
            si2: 0.0,
            si3: 0.0,
            sl2: 0.0,
            sl3: 0.0,
            sl4: 0.0,
            gsto: 0.0,
            xfact: 0.0,
            xgh2: 0.0,
            xgh3: 0.0,
            xgh4: 0.0,
            xh2: 0.0,
            xh3: 0.0,
            xi2: 0.0,
            xi3: 0.0,
            xl2: 0.0,
            xl3: 0.0,
            xl4: 0.0,
            xlamo: 0.0,
            zmol: 0.0,
            zmos: 0.0,
            atime: 0.0,
            xli: 0.0,
            xni: 0.0,

            /* Gravity */
            tumin: 0.0,
            mus: 0.0,
            radiusearthkm: 0.0,
            xke: 0.0,
            j2: 0.0,
            j3: 0.0,
            j4: 0.0,
            j3oj2: 0.0,

            /* Mean elements */
            am: 0.0,
            em: 0.0,
            im: 0.0,
            om: 0.0,
            mm: 0.0,
            nm: 0.0,

            /*Elements */
            a: 0.0,
            altp: 0.0,
            alta: 0.0,
            epochdays: 0.0,
            jdsatepoch: 0.0,
            jdsatepoch_f: 0.0,
            nddot: 0.0,
            ndot: 0.0,
            bstar: 0.0,
            rcse: 0.0,
            inclo: 0.0,
            nodeo: 0.0,
            ecco: 0.0,
            argpo: 0.0,
            mo: 0.0,
            no: 0.0,
            no_kozai: 0.0,
            no_unkozai: 0.0,
        }
    }
}
