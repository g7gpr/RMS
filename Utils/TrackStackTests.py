import os.path
from RMS.Misc import mkdirP
import RMS.ConfigReader as cr

from TrackStack import trackStack

if __name__ == "__main__":
    # ## PARSE INPUT ARGUMENTS ###
    # Init the command line arguments parser

    file_path = os.path.expanduser(("~/tmp/AU0028_20240814_123650_883664/"))

    config = cr.loadConfigFromDirectory(".config", file_path)


    test_list = [0,1,2,-1,'text_option',65536]
    result_dir = os.path.expanduser("~/tmp/trackstacktesting")

    for text_option, dest_folder in zip(test_list, range(0,len(test_list))):

        stack_out_dir = os.path.join(result_dir,str(dest_folder))
        mkdirP(result_dir)
        print("Writing to {:s}".format(result_dir))
        trackStack([file_path], config, background_compensation=False,
            hide_plot=False, showers=None,
            darkbackground=False, out_dir=result_dir, scalefactor=2,
            draw_constellations=True, one_core_free=True,
            textoption=text_option)
