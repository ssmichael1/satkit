# %%

import datetime

import satkit as sk

tm = sk.time(2024, 3, 1, 0, 0, 0.3)
print(tm)


tle_lines = [
    "0 STARLINK-30477",
    "1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997",
    "2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746",
]
tle = sk.TLE.from_lines(tle_lines)
dt = datetime.datetime(2024, 4, 12, 0, 0, 0, 300000)
print(dt)
print(tle.epoch)

p, v = sk.sgp4(tle, dt)
print(p)
print(v)

print(sk.time.GPS_EPOCH)

# %%
