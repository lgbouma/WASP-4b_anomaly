import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os

def main():

    fei_df = pd.read_csv('../data/fei_times.csv')

    mydir = '/home/luke/Dropbox/proj/tessorbitaldecay/data'
    myfname = 'WASP-4b_transits_and_TESS_times_O-C_vs_epoch_selected.csv'

    my_df = pd.read_csv(os.path.join(mydir, myfname), sep=';')
    my_tess = my_df[my_df['original_reference']=='me']

    my_tess_times = np.array(my_tess['sel_transit_times_BJD_TDB'] - 2457000)
    fei_times = np.array(fei_df['BJD-2457000'])

    my_unc = np.array(my_tess['err_sel_transit_times_BJD_TDB'])
    fei_unc = np.array(fei_df['unc'])

    print('luke times\n')
    print(my_tess_times)

    print('\nfei times\n')
    print(fei_times)

    print('\n(luke-fei) times (units: minutes)\n')
    print((my_tess_times - fei_times)*24*60)

    print('\nstatistics for (luke times - fei times), units: minutes\n')
    print(
        pd.DataFrame(
        {'luke_time_minus_fei_time':(my_tess_times - fei_times)*24*60}
        ).describe()
    )

    print('\n(luke-fei) times / fei_unc\n')
    print((my_tess_times - fei_times)/fei_unc)

    print('\n(luke-fei) times / luke_unc\n')
    print((my_tess_times - fei_times)/my_unc)

    print('\nfei unc / luke_unc\n')
    print(fei_unc/my_unc)

    n, bins, patches = plt.hist((my_tess_times - fei_times)*24*60, 10, facecolor='g', alpha=0.75)

    plt.xlabel('(luke time - fei time) [minutes]')
    plt.ylabel('number')

    plt.title('do independent time measurements agree?')

    plt.grid(True)
    savname = '../results/check_times_vs_feis.png'
    plt.savefig(savname, dpi=350,
                bbox_inches='tight')
    print('\nmade {}'.format(savname))

    import IPython; IPython.embed()

if __name__=="__main__":
    main()
