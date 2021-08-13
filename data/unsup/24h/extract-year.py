import pandas as pd
import sys

infile = sys.argv[1]
outfile = sys.argv[2]
year = sys.argv[3]

df = pd.read_csv(infile)
print(df.shape)
df = df[df.created_date.str.startswith(year, na = False)]
print(df.shape)

ok, notok = 0, 0
with open(outfile, "w") as of:

    for t in df.content.tolist():
        try:
            text = t.replace("\n", " ").replace("\r", " ") + "\n"
            text = text.lower()
            of.write(text)
            ok += 1
        except:
            notok += 1

print("Done! %d / %d Ok," % (ok, ok + notok))


