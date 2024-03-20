import satkit as sk
import numpy as np


def read_sp3file(fname, satnum=20):
    """
    Read SP3 file
    (file containing "true" GPS ephemerides)
    and output UTC time and position in ITRF frame
    """

    # Read in the test vectors
    with open(fname, "r") as fd:
        lines = fd.readlines()

    def line2date(lines):
        for line in lines:
            year = int(line[3:7])
            month = int(line[8:10])
            day = int(line[11:13])
            hour = int(line[14:16])
            minute = int(line[17:19])
            sec = float(line[20:32])
            yield sk.time(year, month, day, hour, minute, sec)

    def line2pos(lines):
        for line in lines:
            lvals = line.split()
            yield np.array([float(lvals[1]), float(lvals[2]), float(lvals[3])])

    datelines = list(filter(lambda x: x[0] == "*", lines))
    match = f"PG{satnum:02d}"
    satlines = list(filter(lambda x: x[0:4] == match, lines))
    dates = np.fromiter(line2date(datelines), sk.time)
    pitrf = np.stack(np.fromiter(line2pos(satlines), list), axis=0) * 1.0e3

    return (pitrf, dates)
