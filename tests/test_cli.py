from typer.testing import CliRunner

from pacpac.cli import cli
from test_pacpac import TEST_VH_AA_SEQ, TEST_FILE_NAME, TEST_VH_COL_NAME


runner = CliRunner()

def test_cluster_cli():
    result = runner.invoke(cli, ['cluster', TEST_FILE_NAME, TEST_VH_COL_NAME])
    assert result.exit_code == 0


def test_probe_cli():
    result = runner.invoke(cli, ['probe', TEST_VH_AA_SEQ, TEST_FILE_NAME, TEST_VH_COL_NAME])
    assert result.exit_code == 0
