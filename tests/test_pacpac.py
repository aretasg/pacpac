import unittest

import pandas as pd

from pacpac import pacpac


TEST_VH_AA_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSNYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISTDNARNSLYLQMNSLRAEDTAVYYCARERRGYYYGSGSFSDDYYFGMDVWGQGATVIVSS"
TEST_DATASET = "pertussis_sc_200_head.csv"
TEST_VH_COL_NAME = "VH_AMINO_ACID_SEQUENCE"


class pacpac_test(unittest.TestCase):

    def test_run_and_parse_anarci(self):

        output_dict = pacpac.run_and_parse_anarci(TEST_VH_AA_SEQ)

        self.assertIsInstance(output_dict, dict)
        assert len(output_dict) == 7


    def test_clustering_se_false(self):
        df = pd.read_csv(TEST_DATASET)
        df = pacpac.cluster(
            df,
            TEST_VH_COL_NAME,
            structural_equivalence=False,
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_clustering_se_true(self):
        df = pd.read_csv(TEST_DATASET)
        df = pacpac.cluster(
            df,
            TEST_VH_COL_NAME,
            structural_equivalence=True,
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_clustering_no_clonotyping(self):
        df = pd.read_csv(TEST_DATASET)
        df = pacpac.cluster(
            df,
            TEST_VH_COL_NAME,
            perform_clonotyping=False
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_clustering_tokenize(self):
        df = pd.read_csv(TEST_DATASET)
        df = pacpac.cluster(
            df,
            TEST_VH_COL_NAME,
            tokenize=True
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_probing_ignore_se_false(self):

        df = pd.read_csv(TEST_DATASET)
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            df,
            TEST_VH_COL_NAME,
            structural_equivalence=False,
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_probing_ignore_se_true(self):

        df = pd.read_csv(TEST_DATASET)
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            df,
            TEST_VH_COL_NAME,
            structural_equivalence=True,
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_probing_no_clonotyping(self):

        df = pd.read_csv(TEST_DATASET)
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            df,
            TEST_VH_COL_NAME,
            perform_clonotyping=False
        )

        self.assertIsInstance(df, pd.DataFrame)


    def test_probing_tokenize(self):

        df = pd.read_csv(TEST_DATASET)
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            df,
            TEST_VH_COL_NAME,
            tokenize=True
        )

        self.assertIsInstance(df, pd.DataFrame)


if __name__ == "__main__":

    unittest.main()
