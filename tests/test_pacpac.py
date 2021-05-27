import unittest

import pandas as pd

from pacpac import annotations
from pacpac import pacpac


TEST_VH_AA_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSNYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISTDNARNSLYLQMNSLRAEDTAVYYCARERRGYYYGSGSFSDDYYFGMDVWGQGATVIVSS"
TEST_VL_AA_SEQ = "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDTEST_DFATYYCQQYNSYSLTFGGGTKVEIK"
TEST_VH_COL_NAME = "VH_AMINO_ACID_SEQUENCE"
TEST_VL_COL_NAME = "VL_AMINO_ACID_SEQUENCE"
TEST_DF = pd.read_csv("pertussis_sc_200_head.csv")


class pacpac_test(unittest.TestCase):

    def test_run_and_parse_anarci(self):

        output_dict = annotations.run_and_parse_anarci(TEST_VH_AA_SEQ)

        self.assertIsInstance(output_dict, dict)
        assert len(output_dict) == 7


    def test_clustering_se_false(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            structural_equivalence=False,
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_clustering_se_true(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            structural_equivalence=True,
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_clustering_no_clonotyping(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            perform_clonotyping=False
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_clustering_tokenize(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            tokenize=True
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_clustering_both_chains(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            TEST_VL_COL_NAME
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_probing_ignore_se_false(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            structural_equivalence=False,
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_probing_ignore_se_true(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            structural_equivalence=True,
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_probing_no_clonotyping(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            perform_clonotyping=False
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_probing_tokenize(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            tokenize=True
        )

        self.assertIsInstance(df, pd.DataFrame)

    def test_probing_both_chains(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            TEST_VL_COL_NAME,
            TEST_VL_AA_SEQ
        )

        self.assertIsInstance(df, pd.DataFrame)


if __name__ == "__main__":

    unittest.main()
