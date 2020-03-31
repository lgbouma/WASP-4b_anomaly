import pandas as pd

df = pd.read_csv(
    '../paper/MRTs/WASP-4b_literature_and_TESS_times_O-C_vs_epoch_20200127_selected.csv',
    sep = ';'
)

scols = [
    'sel_transit_times_BJD_TDB',
    'err_sel_transit_times_BJD_TDB',
    'sel_epoch',
    'original_reference',
    'where_I_got_time'
]

df = df[scols]

df['sel_epoch'] = df['sel_epoch'].astype(int)

rdict = {
    'Baluev2019supp':'2019MNRAS.490.1294B',
    'measured_from_SPOC_alert_LC':'2019AJ....157..217B',
    'me':'2019AJ....157..217B',
}
for k,v in rdict.items():
    for c in ['where_I_got_time','original_reference']:
        df[c] = df[c].replace(to_replace=k, value=v)

df.to_csv('../paper/MRTs/transit_data.csv', index=False, float_format='%.6f')
