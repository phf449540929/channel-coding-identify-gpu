@echo off

rem 我的天，bat对空格敏感，=前后不能有空格

rem set filter_number_array=2 3 4 5 6
rem set filter_size_array=16 32

rem set snr_array=30 28 26 24 22 20 18 16 14 12 10 8 6 4 2 0
rem set epoch_array=100 200 300 400 500 600 700 800 900 1000

set snr_array=30 28 26 24 22 20 18 16 14 12 10 8 6 4 2 0
set epoch_array=100 200 300 400 500 600 700 800 900 1000

rem filter_size_array=64

for %%a in (%snr_array%) do (
    for  %%b in (%epoch_array%) do (
        python DRSN_TFLearn-Turbo.py %%a %%b
    )
)

rem for %%a in (%filter_number_array%) do (
rem     for  %%b in (%filter_size_array%) do (
rem         python DRSN_TFLearn.py %%a %%b
rem     )
rem )
