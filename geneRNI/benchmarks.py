import os
import pathlib
import sys

import numpy as np
import pandas as pd

from geneRNI.tools import Data, Settings

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(), '..')
sys.path.insert(0, dir_main)


class Benchmark:

    @staticmethod
    def f_golden_dream4(size: int, network: int) -> pd.DataFrame:
        """ retrieves golden links for dream4 for given size and network """
        dir_ = os.path.join(dir_main,
                            f'data/dream4/gold_standards/{size}/DREAM4_GoldStandard_InSilico_Size{size}_{network}.tsv')
        return pd.read_csv(dir_, names=['Regulator', 'Target', 'Weight'], sep='\t')

    @staticmethod
    def f_data_dream4(size: int, network: int):
        """ retrieves train data for dream4 for given size and network"""
        (TS_data, time_points, SS_data) = pd.read_pickle(
            os.path.join(dir_main, f'data/dream4/data/size{size}_{network}_data.pkl'))
        gene_names = [f'G{i}' for i in range(1, size + 1)]
        return TS_data, time_points, SS_data, gene_names

    @staticmethod
    def f_golden_dream5(network: int) -> pd.DataFrame:
        """ retrieves golden links for dream5 for given network """
        if network == 1:
            filename = 'dream5_networkinference_goldstandard_network1 - in silico.tsv'
        elif network == 3:
            filename = 'dream5_networkinference_goldstandard_network3 - e. coli.tsv'
        elif network == 4:
            filename = 'dream5_networkinference_goldstandard_network4 - s. cerevisiae.tsv'
        else:
            raise NotImplementedError(f'Unknown network "{network}" in DREAM5 dataset')
        dir_ = os.path.join(dir_main, 'data', 'dream5', 'testData', filename)
        return pd.read_csv(dir_, names=['Regulator', 'Target', 'Weight'], sep='\t')

    @staticmethod
    def f_data_dream5(network: int):
        """ retrieves train data for dream5 for network"""
        data = pd.read_csv(os.path.join(dir_main, f'data/dream5/trainingData/net{network}_expression_data.tsv'),
                           sep='\t')
        transcription_factors = pd.read_csv(
            os.path.join(dir_main, f'data/dream5/trainingData/net{network}_transcription_factors.tsv'), sep='\t',
            header=None)
        gene_names = data.columns.tolist()
        SS_data = data.values
        return SS_data, gene_names, transcription_factors[0].tolist()

    @staticmethod
    def f_data_melanogaster():
        """ retrieves train data for melanogaster"""
        (TS_data, time_points, genes, TFs, alphas) = pd.read_pickle(
            os.path.join(dir_main, f'data/real_networks/data/drosophila_data.pkl'))
        return TS_data, time_points, genes, TFs, alphas

    @staticmethod
    def f_data_ecoli():
        """ retrieves train data for ecoli"""
        (TS_data, time_points, genes, TFs, alphas) = pd.read_pickle(
            os.path.join(dir_main, f'data/real_networks/data/ecoli_data.pkl'))
        return TS_data, time_points, genes, TFs, alphas

    @staticmethod
    def f_data_cerevisiae():
        """ retrieves train data for yeast"""
        (TS_data, time_points, genes, TFs, alphas) = pd.read_pickle(
            os.path.join(dir_main, f'data/real_networks/data/cerevisiae_data.pkl'))
        return TS_data, time_points, genes, TFs, alphas

    @staticmethod
    def f_data_GRN(method, noise_level, network):
        """ retrieves train data for GRNbenchmark for given specs"""
        dir_data_benchmark = os.path.join(dir_main, 'data/GRNbenchmark')
        base = method + '_' + noise_level + '_' + network
        file_exp = base + '_' + 'GeneExpression.csv'
        file_per = base + '_' + 'Perturbations.csv'
        file_exp = os.path.join(dir_data_benchmark, file_exp)
        file_per = os.path.join(dir_data_benchmark, file_per)

        exp_data = pd.read_csv(file_exp)
        per_data = pd.read_csv(file_per)

        gene_names = exp_data['Row'].tolist()
        exp_data = np.array([exp_data[col].tolist() for col in exp_data.columns if col != 'Row'])
        per_data = [gene_names[per_data[col].tolist().index(-1)] for col in per_data.columns if col != 'Row']
        return exp_data, per_data, gene_names

    @staticmethod
    def process_data(ts_data, ss_data, time_points, gene_names, estimator_t, **specs) -> Data:
        pp = Settings.default(estimator_t)
        return Data(
            gene_names,
            ss_data,
            ts_data,
            time_points,
            test_size=pp.test_size,
            random_state=pp.random_state_data,
            **specs
        )

    @staticmethod
    def process_data_dream4(size, network, estimator_t: str, **specs) -> Data:
        ts_data, time_points, ss_data, gene_names = Benchmark.f_data_dream4(size, network)
        return Benchmark.process_data(ts_data, ss_data, time_points, gene_names, estimator_t, **specs)

    @staticmethod
    def process_data_dream5(network, estimator_t: str, **specs) -> Data:
        ss_data, gene_names, tf_names = Benchmark.f_data_dream5(network)
        return Benchmark.process_data(
            None, ss_data, None, gene_names, estimator_t, regulators=tf_names, **specs)

    @staticmethod
    def process_data_grn_benchmark(method, noise_level, network, estimator_t: str, **specs) -> Data:
        ss_data, _, gene_names = Benchmark.f_data_GRN(method, noise_level, network)
        return Benchmark.process_data(None, ss_data, None, gene_names, estimator_t)
