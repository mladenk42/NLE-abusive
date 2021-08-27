import pandas as pd
import sys

infile = sys.argv[1]
outfile = sys.argv[2]
s_year = sys.argv[3]
e_year = sys.argv[4]

df = pd.read_csv(infile)
print(df.shape)


of = open(outfile, "w")

for year in range(s_year,e_year+1):
    df = df[df.created_date.str.startswith(year, na = False)]
    print(df.shape)

    ok, notok = 0, 0

    for t in df.content.tolist():
        try:
            text = t.replace("\n", " ").replace("\r", " ") + "\n"
            of.write(text)
            ok += 1
        except:
            notok += 1

print("Done! %d / %d %s Ok," % (ok, ok + notok, outfile))
of.close()

