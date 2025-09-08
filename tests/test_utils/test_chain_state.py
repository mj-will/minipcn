import matplotlib.figure
import pytest

from minipcn.utils import ChainState, ChainStateHistory


@pytest.fixture
def sample_chain_states():
    return [
        ChainState(
            it=0,
            acceptance_rate=0.1,
            target_acceptance_rate=0.2,
            step="step1",
            extra_stats={"energy": 1.0},
        ),
        ChainState(
            it=1,
            acceptance_rate=0.3,
            target_acceptance_rate=0.2,
            step="step1",
            extra_stats={"energy": 1.5},
        ),
        ChainState(
            it=2,
            acceptance_rate=0.5,
            target_acceptance_rate=0.2,
            step="step1",
            extra_stats={"energy": 2.0},
        ),
    ]


def test_chain_state_creation():
    cs = ChainState(
        it=5,
        acceptance_rate=0.4,
        target_acceptance_rate=0.3,
        step="abc",
        extra_stats={"foo": 42},
    )
    assert cs.it == 5
    assert cs.acceptance_rate == 0.4
    assert cs.target_acceptance_rate == 0.3
    assert cs.step == "abc"
    assert cs.extra_stats["foo"] == 42


def test_chain_state_history_from_chain_states(sample_chain_states):
    history = ChainStateHistory.from_chain_states(sample_chain_states)

    assert history.it == [0, 1, 2]
    assert history.acceptance_rate == [0.1, 0.3, 0.5]
    assert history.target_acceptance_rate == [0.2, 0.2, 0.2]
    assert history.extra_stats == {"energy": [1.0, 1.5, 2.0]}


def test_chain_state_history_getitem_index(sample_chain_states):
    history = ChainStateHistory.from_chain_states(sample_chain_states)
    single = history[1]

    assert isinstance(single, ChainStateHistory)
    assert single.it == [1]
    assert single.acceptance_rate == [0.3]
    assert single.extra_stats == {"energy": [1.5]}


def test_chain_state_history_getitem_slice(sample_chain_states):
    history = ChainStateHistory.from_chain_states(sample_chain_states)
    sliced = history[1:]

    assert sliced.it == [1, 2]
    assert sliced.acceptance_rate == [0.3, 0.5]
    assert sliced.extra_stats == {"energy": [1.5, 2.0]}


def test_chain_state_history_getitem_invalid_type(sample_chain_states):
    history = ChainStateHistory.from_chain_states(sample_chain_states)
    with pytest.raises(TypeError):
        _ = history["not an index"]


@pytest.mark.usefixtures("close_figures")
def test_plot_acceptance_rate(sample_chain_states):
    history = ChainStateHistory.from_chain_states(sample_chain_states)
    fig = history.plot_acceptance_rate()

    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.usefixtures("close_figures")
def test_plot_extra_stat(sample_chain_states):
    history = ChainStateHistory.from_chain_states(sample_chain_states)
    fig = history.plot_extra_stat("energy")

    assert isinstance(fig, matplotlib.figure.Figure)
