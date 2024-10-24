use crate::utils::{self, download_if_not_exist, SKResult};
use nalgebra as na;
use std::io::{self, BufRead};
use std::path::PathBuf;

#[derive(Debug)]
pub struct IERSTable {
    data: [na::DMatrix<f64>; 6],
}

impl IERSTable {
    pub fn from_file(fname: &str) -> SKResult<IERSTable> {
        let mut table = IERSTable {
            data: [
                na::DMatrix::<f64>::zeros(0, 0),
                na::DMatrix::<f64>::zeros(0, 0),
                na::DMatrix::<f64>::zeros(0, 0),
                na::DMatrix::<f64>::zeros(0, 0),
                na::DMatrix::<f64>::zeros(0, 0),
                na::DMatrix::<f64>::zeros(0, 0),
            ],
        };

        let path = utils::datadir().unwrap_or(PathBuf::from(".")).join(fname);
        download_if_not_exist(&path, None)?;

        let mut tnum: i32 = -1;
        let mut rowcnt: usize = 0;
        let file = std::fs::File::open(&path)?;
        let lines = io::BufReader::new(file).lines();

        for line in lines {
            match line {
                Ok(l) => {
                    let tline = l.trim();
                    if tline.len() < 10 {
                        continue;
                    }
                    if tline[..3].eq("j =") {
                        tnum = tline[4..5].parse()?;
                        let s: Vec<&str> = tline.split_whitespace().collect();
                        let tsize: usize = s[s.len() - 1].parse().unwrap_or(0);
                        if !(0..=5).contains(&tnum) || tsize == 0 {
                            return utils::skerror!(
                                "Error parsing file {}, invalid table definition line",
                                fname
                            );
                        }
                        table.data[tnum as usize] = na::DMatrix::<f64>::zeros(tsize, 17);
                        rowcnt = 0;
                        continue;
                    } else if tnum >= 0 {
                        if table.data[tnum as usize].ncols() < 17 {
                            return Err(utils::SKErr::new(
                                format!("Error parsing file {}, table not initialized", fname)
                                    .as_str(),
                            )
                            .into());
                        }
                        table.data[tnum as usize].set_row(
                            rowcnt,
                            &na::SMatrix::<f64, 1, 17>::from_iterator(
                                tline
                                    .split_whitespace()
                                    .map(|x| x.parse().unwrap()),
                            ),
                        );
                        rowcnt += 1;
                    }
                }
                Err(_) => continue,
            }
        }
        Ok(table)
    }

    pub fn compute(&self, t_tt: f64, delaunay: &na::SVector<f64, 14>) -> f64 {
        let mut retval: f64 = 0.0;
        for i in 0..6 {
            // return if finished
            if self.data[i].ncols() == 0 {
                continue;
            }

            let mut tmult: f64 = 1.0;
            for _ in 0..i {
                tmult *= t_tt;
            }

            for j in 0..self.data[i].nrows() {
                //double argVal = 0;
                let mut argval: f64 = 0.0;
                for k in 0..13 {
                    argval += self.data[i][(j, k + 3)] * delaunay[k];
                }
                let sval = f64::sin(argval);
                let cval = f64::cos(argval);
                retval += tmult * (self.data[i][(j, 1)] * sval + self.data[i][(j, 2)] * cval);
            }
        }
        retval
    }
}

#[cfg(test)]
mod tests {
    use super::IERSTable;

    #[test]
    fn load_table() {
        let t = IERSTable::from_file("tab5.2a.txt");
        if t.is_err() {
            panic!("Could not load IERS table");
        }
        if t.unwrap().data[0].ncols() < 17 {
            panic!("Error loading table");
        }
    }
}
