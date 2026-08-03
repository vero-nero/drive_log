OBD DRIVE LOG ANALYZER
======================

Requirements
------------
- Python 3.10 or newer
- pandas
- numpy
- matplotlib
- Tkinter, included in the standard Windows Python installer

Installation on Windows
-----------------------
Open Command Prompt in this folder and run:

    py -m pip install -r requirements.txt

Start the graphical viewer
--------------------------

    py obd_log_analyzer.py

You can also open a file immediately:

    py obd_log_analyzer.py obd_log.csv

Create an HTML report without opening the graphical viewer
-----------------------------------------------------------

    py obd_log_analyzer.py obd_log.csv --report obd_report.html

Optional atmospheric pressure setting
-------------------------------------
The program calculates relative boost from absolute MAP when the CSV boost
column is empty. The default atmospheric pressure is 101.3 kPa.

    py obd_log_analyzer.py obd_log.csv --report obd_report.html --atmospheric-kpa 95.0

Main functions
--------------
- Automatic separator and common encoding detection
- Automatic matching of common OBD sensor column names
- Overview cards for duration, distance, sampling rate and maximum values
- Restrained, readable chart presets
- Configurable smoothing
- Automatic boost calculation from MAP
- Fuel-trim, temperature, RPM, boost, MIL/DTC and loaded-lambda checks
- Consecutive warning samples grouped into event ranges
- Data-quality checks for slow/irregular logging and constant sensors
- Raw-data table
- Self-contained HTML report export

Important interpretation note
-----------------------------
A slow logger can miss boost spikes, knock events and short lean conditions.
The supplied example logs approximately one sample every 10 seconds. Use a
faster logging method for full-load diagnosis.
