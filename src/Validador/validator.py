from io import BytesIO
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error
import subprocess
import sys

INP_FILE = 'sample_in.csv'
OUT_FILE = 'sample_out.csv'


def test (cmd, file, real):
    """
    Tests the output of a ML command.

    @param cmd: the command to evaluate.
    @param file: the datafile to introduce.
    @param real: the real output.
    @return a score of the tested command.
    """
    output = subprocess.check_output([sys.executable, cmd, file])
    csvStr = BytesIO(output)
    data = pd.read_csv(csvStr)
    auc_score = roc_auc_score(real.CreditoAprobado, data.CreditoAprobado)
    mae_score = mean_absolute_error(real.ScoreRiesgo, data.ScoreRiesgo)
    return auc_score, mae_score


if __name__ == '__main__':
    real = pd.read_csv(OUT_FILE)
    for folder in filter(lambda f : os.path.isdir(f), os.listdir('.')):
        try:
            score_clf, score_ref = test(f'{folder}/main.py', INP_FILE, real)
            print(f'{folder}: {score_clf:.3f} / {score_ref:.3f}')
        except Exception as e:
            print(f'Ha ocurrido un error con {folder}: {e}')