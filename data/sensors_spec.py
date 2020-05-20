'''
File to set sensors specifications
such as noise equivalent irradiance/radiance (NEI), fov, etc.
NEI in µW m-2 nm-1 (sr-1)
units:
irradiance: mW m-2 nm-1
radiance: mW m-2 nm-1 sr-1
depth: m
'''

spec=dict(
    radiance=dict(
            noise=0.25)
)
delta_Lu_depth=0.07
delta_Edz_depth=-0.28
