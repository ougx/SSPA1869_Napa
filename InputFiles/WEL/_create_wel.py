import pandas as pd
import numpy as np
import flopy

nw1 = 7614

def mf_read_dis_time(dis_file, nskip=0, startdate='1990-01-01'):
    """
    create a time index based on the DIS file
    """
    dis_time = pd.read_table(
        dis_file,
        skiprows=nskip,
        sep="\\s+",
        usecols=[0,1,2,3],
        names='PERLEN NSTP TSMULT SsTr'.split())
    #nstep = dis_time['NSTP']
    time = []
    dtt = []
    cumtim = 0
    period = []
    step = []

    for i, row in dis_time.iterrows():
        if row['TSMULT'] == 1:
            dt = row['PERLEN'] / row['NSTP']
        else:
            dt = row['PERLEN'] * (row['TSMULT'] - 1) / (
                row['TSMULT']**row['NSTP'] - 1)
        for stp in range(row['NSTP']):
            period.append(i+1)
            step.append(stp+1)
            cumtim = cumtim + dt
            dtt.append(dt)
            time.append(cumtim)
            dt = dt * row['TSMULT']
    return pd.DataFrame({'Period':period,'Step':step, 'Dt': dtt}, index=pd.to_datetime(startdate) + pd.to_timedelta(time, 'D'),)


def mf_split_wel(welfile, skipline=0, prefix='', startdate='2000-01-01', disfile='', disskip_per=0):
    # read the dates from DIS file
    times = mf_read_dis_time(disfile, nskip=disskip_per, startdate=startdate)
    periods = times.reset_index().groupby('Period').last()['index']
    with open(prefix + 'main.wel', 'w') as fout:
        with open(welfile, 'r') as fwel:
            for i in range(skipline):
                fout.write(fwel.readline())
            rate0 = ""
            for t in periods:
                litmp = fwel.readline()
                itmp = int(litmp.split()[0])
                rates = "".join([fwel.readline() for i in range(itmp)])
                if rate0 == rates:
                    fout.write("-1\n")
                else:
                    fname = prefix + t.strftime("%Y%m") + '.wel'
                    print(fname, itmp)
                    fout.write(litmp)
                    fout.write('OPEN/CLOSE ' + fname + '\n')
                    with open(fname, 'w') as ff:
                        ff.write(rates)
                        rate0 = rates


welfile = r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\NVIHM_SWRCB\InputFiles\WEL\WEL.wel"
skipline = 4
prefix = "WEL_"
startdate = "1984-03-31"
dis_file = "..\DIS\DIS.dis"
disskip_per = 17
nskip = disskip_per
mf_split_wel(welfile, skipline, prefix, startdate, dis_file, disskip_per)
