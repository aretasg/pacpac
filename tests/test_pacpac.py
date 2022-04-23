import unittest

import pandas as pd

from pacpac import annotations
from pacpac import pacpac


# CL-94726
TEST_VH_AA_SEQ = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGRNKYYADSVKGRFTISRDNSKNMLYLQMNSLRAEDTGVYYCARDHGILTGYSSRFDYWGQGTLVTVSS"
TEST_VL_AA_SEQ = "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNSYSLTFGGGTKVEIK"
TEST_VH_COL_NAME = "VH_AMINO_ACID_SEQUENCE"
TEST_VL_COL_NAME = "VL_AMINO_ACID_SEQUENCE"
TEST_FILE_NAME = "pertussis_sc_300_head.csv"
TEST_DF = pd.read_csv(TEST_FILE_NAME)


class pacpac_test(unittest.TestCase):

    def test_run_and_parse_anarci(self):

        output_dict = annotations.run_and_parse_anarci(TEST_VH_AA_SEQ)

        self.assertIsInstance(output_dict, dict)
        assert len(output_dict) == 7


    def test_parapred(self):

        expected_output_dict = {
            "CDR1": [
                (0, "A", 0.008999248),
                (1, "S", 0.06711262),
                (2, "G", 0.2511713),
                (3, "F", 0.39113072),
                (4, "T", 0.5397705),
                (5, "F", 0.06678333),
                (6, "S", 0.6621542),
                (7, "S", 0.88582015),
                (8, "Y", 0.7745048),
                (9, "G", 0.63607603),
                (10, "M", 0.047867972),
            ],
            "CDR2": [
                (0, "V", 0.40129197),
                (1, "I", 0.08517706),
                (2, "W", 0.890859),
                (3, "Y", 0.64144516),
                (4, "D", 0.82870287),
                (5, "G", 0.42383772),
                (6, "R", 0.755072),
                (7, "N", 0.80309033),
                (8, "K", 0.4333197),
                (9, "Y", 0.8859638),
            ],
            "CDR3": [
                (0, "A", 0.013490887),
                (1, "R", 0.42270353),
                (2, "D", 0.5987166),
                (3, "H", 0.8661246),
                (4, "G", 0.9240452),
                (5, "I", 0.9616318),
                (6, "L", 0.97830427),
                (7, "T", 0.9071267),
                (8, "G", 0.9255135),
                (9, "Y", 0.94964755),
                (10, "S", 0.8045249),
                (11, "S", 0.7637215),
                (12, "R", 0.505665),
                (13, "F", 0.042653713),
                (14, "D", 0.19786143),
                (15, "Y", 0.12871934),
                (16, "W", 0.0050766855),
                (17, "G", 0.009074119),
            ],
        }

        cdrs = annotations.get_annotations(TEST_VH_AA_SEQ, scheme="imgt",
            cdr_scheme="chothia", num_extra_residues=2)
        output_dict = annotations.get_paratope_probabilities(cdrs)

        import math

        same_prediction = True
        for cdr in ['CDR1', 'CDR2', 'CDR3']:
            for index, res in enumerate(output_dict[cdr]):
                if res[1] == expected_output_dict[cdr][index][1] and \
                    math.isclose(res[2], expected_output_dict[cdr][index][2], rel_tol=1e-5):
                    pass
                else:
                    same_prediction = False
                    break

        self.assertIsInstance(output_dict, dict)
        assert len(output_dict) == 3
        assert same_prediction is True


    def test_clustering_se_false(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            structural_equivalence=False,
        )

        self.assertIsInstance(df, pd.DataFrame)
        assert df['PARATOPE_CLUSTER'].notna().sum() == 140


    def test_clustering_se_true(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            structural_equivalence=True,
        )

        predicted_paratopes = df['PARATOPE'].tolist()
        expected_paratopes = TEST_DF['EXPECTED_PARATOPE'].tolist()

        self.assertIsInstance(df, pd.DataFrame)
        assert df['PARATOPE_CLUSTER'].notna().sum() == 128
        assert df['CLONOTYPE_CLUSTER'].notna().sum() == 130
        assert predicted_paratopes == expected_paratopes


    def test_clustering_no_clonotyping(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            perform_clonotyping=False
        )

        is_clonotype_column = False
        if 'CLONOTYPE_CLUSTER' in list(df.columns):
            is_clonotype_column = True
        self.assertIsInstance(df, pd.DataFrame)
        assert is_clonotype_column is False


    def test_clustering_tokenize(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            tokenize=True
        )

        self.assertIsInstance(df, pd.DataFrame)
        assert df['PARATOPE_CLUSTER'].notna().sum() == 145


    def test_clustering_both_chains(self):
        df = pacpac.cluster(
            TEST_DF,
            TEST_VH_COL_NAME,
            TEST_VL_COL_NAME
        )

        self.assertIsInstance(df, pd.DataFrame)
        assert df['PARATOPE_CLUSTER'].notna().sum() == 116


    def test_probing_ignore_se_false(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            structural_equivalence=False,
        )

        both_count = len(df.loc[df['PREDICTION_SPACE'] == 'both'])
        paratope_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'paratope-only'])
        clonotype_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'clonotype-only'])

        self.assertIsInstance(df, pd.DataFrame)
        assert both_count == 2
        assert paratope_only_count == 2
        assert clonotype_only_count == 0


    def test_probing_ignore_se_true(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            structural_equivalence=True,
        )

        both_count = len(df.loc[df['PREDICTION_SPACE'] == 'both'])
        paratope_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'paratope-only'])
        clonotype_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'clonotype-only'])

        self.assertIsInstance(df, pd.DataFrame)
        assert both_count == 2
        assert paratope_only_count == 1
        assert clonotype_only_count == 0


    def test_probing_no_clonotyping(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            perform_clonotyping=False
        )

        both_count = len(df.loc[df['PREDICTION_SPACE'] == 'both'])
        paratope_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'paratope-only'])
        clonotype_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'clonotype-only'])

        self.assertIsInstance(df, pd.DataFrame)
        assert both_count == 0
        assert paratope_only_count == 3
        assert clonotype_only_count == 0


    def test_probing_tokenize(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            tokenize=True
        )

        both_count = len(df.loc[df['PREDICTION_SPACE'] == 'both'])
        paratope_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'paratope-only'])
        clonotype_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'clonotype-only'])

        self.assertIsInstance(df, pd.DataFrame)
        assert both_count == 2
        assert paratope_only_count == 2
        assert clonotype_only_count == 0


    def test_probing_both_chains(self):
        df = pacpac.probe(
            TEST_VH_AA_SEQ,
            TEST_DF,
            TEST_VH_COL_NAME,
            TEST_VL_COL_NAME,
            TEST_VL_AA_SEQ
        )

        both_count = len(df.loc[df['PREDICTION_SPACE'] == 'both'])
        paratope_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'paratope-only'])
        clonotype_only_count = len(df.loc[df['PREDICTION_SPACE'] == 'clonotype-only'])

        self.assertIsInstance(df, pd.DataFrame)
        assert both_count == 2
        assert paratope_only_count == 2
        assert clonotype_only_count == 0


    def test_probing_multiple_ignore_se_false(self):
        df = pacpac.probe_multiple(
            pd.DataFrame([TEST_VH_AA_SEQ], columns=[TEST_VH_COL_NAME]),
            TEST_DF,
            TEST_VH_COL_NAME,
            structural_equivalence=False,
        )[0]

        both_count = len(df[(df["PARATOPE_MATCH_0"] >= 0.75) & (df["CLONOTYPE_MATCH_0"] >= 0.72)])
        paratope_only_count = len(df[(df["PARATOPE_MATCH_0"] >= 0.75) & (df["CLONOTYPE_MATCH_0"] < 0.72)])
        clonotype_only_count = len(df[(df["PARATOPE_MATCH_0"] < 0.75) & (df["CLONOTYPE_MATCH_0"] >= 0.72)])

        self.assertIsInstance(df, pd.DataFrame)
        assert both_count == 2
        assert paratope_only_count == 2
        assert clonotype_only_count == 0


    def test_probing_multiple_both_chains(self):
        df = pacpac.probe(
            pd.DataFrame({TEST_VH_COL_NAME: [TEST_VH_AA_SEQ], TEST_VL_COL_NAME: [TEST_VL_AA_SEQ]}),
            TEST_DF,
            TEST_VH_COL_NAME,
            TEST_VL_COL_NAME,
        )[0]

        both_count = len(df[(df["PARATOPE_MATCH_0"] >= 0.75) & (df["CLONOTYPE_MATCH_0"] >= 0.72)])
        paratope_only_count = len(df[(df["PARATOPE_MATCH_0"] >= 0.75) & (df["CLONOTYPE_MATCH_0"] < 0.72)])
        clonotype_only_count = len(df[(df["PARATOPE_MATCH_0"] < 0.75) & (df["CLONOTYPE_MATCH_0"] >= 0.72)])

        self.assertIsInstance(df, pd.DataFrame)
        assert both_count == 2
        assert paratope_only_count == 2
        assert clonotype_only_count == 0


if __name__ == "__main__":

    unittest.main()
