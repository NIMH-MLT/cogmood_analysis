from cogmood_analysis import survey_helpers as sh
from pytest import raises
import polars as pl
from pathlib import Path
import json


def test_yn_code():
    res = sh.yn_code("jubba", "Yes")
    assert res["jubba"] == True

    res = sh.yn_code("jubba", "No")
    assert res["jubba"] == False

    res = sh.yn_code("jubba", "Prefer not to answer")
    assert res["jubba"] is None

    res = sh.yn_code("jubba", None)
    assert res["jubba"] is None

    res = sh.yn_code("jubba", "Not applicable")
    assert res["jubba"] is None

    with raises(ValueError):
        sh.yn_code("jubba", "3")


def test_str_code():
    res = sh.str_code("jubba", "wubba")
    assert res["jubba"] == "wubba"

    with raises(ValueError):
        sh.str_code("jubba", None)


def test_num_code():
    res = sh.num_code("jubba", "32.3")
    assert res["jubba"] == 32.3

    res = sh.num_code("jubba", None)
    assert res["jubba"] is None

    with raises(ValueError):
        sh.num_code("jubba", "wubba")


def test_likert_code():
    res = sh.likert_code("jubba", "3: many")
    assert res["jubba"] == 3

    with raises(ValueError):
        sh.likert_code("jubba", "wubba")

    with raises((ValueError, AttributeError)):
        sh.likert_code("jubba", None)


def test_ohe_fac():
    choices = ["todaygad7__5", "todayphq8__1", "todayphq8__3"]
    ohea = sh.ohe_fac(choices)
    choices = ["Attention-Deficit/Hyperactivity Disorder (ADHD)", "One-on-one"]
    oheb = sh.ohe_fac(choices, other=True, none=True, force_list=True)
    res = ohea("jubba", ["todayphq8__3"])
    assert res["jubba__todayphq8_3"] == True
    assert res["jubba__todayphq8_1"] == False
    assert res["jubba__todaygad7_5"] == False

    res = oheb("jubba", ["Attention-Deficit/Hyperactivity Disorder (ADHD)"])
    assert res["jubba__attention_deficit_hyperactivity_disorder_adhd_"] == True
    assert res["jubba__one_on_one"] == False
    assert res["jubba__other"] == False
    assert res["jubba__none"] == False

    res = ohea("jubba", ["todayphq8__3", "todayphq8__1"])
    assert res["jubba__todayphq8_3"] == True
    assert res["jubba__todayphq8_1"] == True
    assert res["jubba__todaygad7_5"] == False

    res = oheb("jubba", "Attention-Deficit/Hyperactivity Disorder (ADHD)")
    assert res["jubba__attention_deficit_hyperactivity_disorder_adhd_"] == True
    assert res["jubba__one_on_one"] == False
    assert res["jubba__other"] == False
    assert res["jubba__none"] == False

    res = ohea("jubba", None)
    assert res["jubba__todayphq8_3"] == False
    assert res["jubba__todayphq8_1"] == False
    assert res["jubba__todaygad7_5"] == False

    res = oheb("jubba", "none")
    assert res["jubba__attention_deficit_hyperactivity_disorder_adhd_"] == False
    assert res["jubba__one_on_one"] == False
    assert res["jubba__other"] == False
    assert res["jubba__none"] == True

    res = oheb("jubba", "other")
    assert res["jubba__attention_deficit_hyperactivity_disorder_adhd_"] == False
    assert res["jubba__one_on_one"] == False
    assert res["jubba__other"] == True
    assert res["jubba__none"] == False

    with raises(ValueError):
        ohea("jubba", "wubba")


def test_survey_matrix():
    res = sh.survey_matrix("jubba", {"foo": 1, "bar": "wubba"})
    assert res["foo"] == 1
    assert res["bar"] == "wubba"


def test_extract_resp():
    response = pl.DataFrame(
        sh.extract_responses(
            json.loads(
                (
                    Path(__file__).parent / "test_data/surveyexpectedoutput_1.json"
                ).read_text()
            ),
            decoders=sh.SURVEY_DECODE,
        )
    )
    expected = pl.read_parquet(
        Path(__file__).parent / "test_data/parsed_survey1.parquet"
    )
    assert response.equals(expected)
