import pandas as pd
import sys
import datetime

infile = sys.argv[1]
outfile = sys.argv[2]
s_year = sys.argv[3]
e_year = sys.argv[4]


df = pd.read_csv(infile,  lineterminator='\n', parse_dates=['created_date', 'last_change'])
print(df.shape)

df = df.dropna(subset = ["created_date"])
print(df.shape)

ok, notok = 0, 0


with  open(outfile, "w") as of:
    for year in range(int(s_year),int(e_year)+1):
        df_year = df[(df.created_date >= datetime.datetime(year = year, month = 1, day = 1)) & (df.created_date <= datetime.datetime(year = year, month = 12, day = 31))]
        print(year, df_year.shape)

        for t in df_year.content.tolist():
            try:
                text = t.replace("\n", " ").replace("\r", " ") + "\n"
                of.write(text)
                ok += 1
            except:
                notok += 1

print("Done! %d / %d %s Ok," % (ok, ok + notok, outfile))

