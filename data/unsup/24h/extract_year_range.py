import pandas as pd
import sys

infile = sys.argv[1]
outfile = sys.argv[2]
s_year = sys.argv[3]
e_year = sys.argv[4]

print(type(s_year))

df = pd.read_csv(infile,  lineterminator='\n', parse_dates=['created_date', 'last_change'])
print(df.shape)


of = open(outfile, "w")
ok, notok = 0, 0

for year in range(int(s_year),int(e_year)+1):
    df = df[df.created_date.str.startswith(year, na = False)]
    print(year, df.shape)


    for t in df.content.tolist():
        try:
            text = t.replace("\n", " ").replace("\r", " ") + "\n"
            of.write(text)
            ok += 1
        except:
            notok += 1

print("Done! %d / %d %s Ok," % (ok, ok + notok, outfile))
of.close()

