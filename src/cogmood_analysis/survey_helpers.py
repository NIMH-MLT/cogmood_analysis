from collections.abc import Callable
from copy import copy
from typing import Any


def yn_code(field: str, resp: str | None) -> dict[str, bool | None]:
    """Transform yes-no fields to boolean responses
    Parameters
    ----------
    field : str
        Name of the field
    resp : str or None
        Response

    Returns
    -------
    result : dict
        dictionary with field as key and coded response as value

    """
    if resp == "Yes":
        return {field: True}
    elif resp == "No":
        return {field: False}
    elif resp == "Prefer not to answer":
        return {field: None}
    elif resp is None:
        return {field: None}
    elif "Not applicable" in resp:
        return {field: None}
    else:
        raise ValueError(
            f"Expected Yes, No, Perfer not to Answer, None or Not applicable, but received {resp}"
        )


def str_code(field: str, resp: str) -> dict[str, str]:
    """Passthrough transform for string fields.
    Parameters
    ----------
    field : str
        Name of the field
    resp : str
        Response

    Returns
    -------
    result : dict
        dictionary with field as key and coded response as value
    """
    if isinstance(resp, str):
        return {field: resp}
    else:
        raise ValueError(
            f"expected resp to be a string, but received {resp} of type {type(resp)}"
        )


def num_code(field: str, resp: str | None) -> dict[str, float | None]:
    """Transform for numeric fields.
    Parameters
    ----------
    field : str
        Name of the field
    resp : str
        Response

    Returns
    -------
    result : dict
        dictionary with field as key and coded response as value
    """
    try:
        if resp is None:
            return {field: None}
        return {field: float(resp)}
    except:
        raise ValueError(f"Someting failed when trying to convert {resp} to a float")


def likert_code(field: str, resp: str | None) -> dict[str, float | int]:
    """Transform for likert filds that takes the number from before the colon if present
    Parameters
    ----------
    field : str
        Name of the field
    resp : str
        Response

    Returns
    -------
    result : dict
        dictionary with field as key and coded response as value
    """
    return {field: int(resp.split(":")[0])}


def ohe_fac(
    choices: list[str],
    other: bool = False,
    none: bool = False,
    force_list: bool = False,
) -> Callable[[str, str | list[str]], dict[str, bool]]:
    """Factory for transformers one hot encoding multi-select checkbox respones.
    Parameters
    ----------
    choices : list
        List of choices
    other : bool
        Add an other choice
    none : bool
        Add a none choice
    force_list : bool
        returned transformer forces the response to be a list

    Returns
    -------
    ohe : function
        transformer function for one hot encoding the field
    """
    choices = copy(choices)
    if other:
        choices.append("other")
    if none:
        choices.append("none")

    def ohe(field: str, resp: str | list[str]) -> dict[str, bool]:
        if resp is None:
            resp = []
        if not isinstance(resp, list):
            if force_list:
                resp = [resp]
            else:
                raise ValueError(f"Expected resp to be a list, but received {resp}.")
        ohe_resp = {}
        for choice in choices:
            clean_choice = (
                choice.lower()
                .replace("-", "_")
                .replace(":", "_")
                .replace(",", "_")
                .replace(" ", "_")
                .replace("(", "_")
                .replace(")", "_")
                .replace("/", "_")
                .replace("__", "_")
                .replace("__", "_")
            )
            ohe_resp[f"{field}__{clean_choice}"] = choice in resp
        return ohe_resp

    return ohe


def survey_matrix(
    field: str, resp: dict[str, str | float | int | None]
) -> dict[str, str | float | int | None]:
    """Simple transformer that unpacks matrix respones
    Paramters
    ---------
    field : str
    resp : dict

    Returns
    -------
    resp : dict
    """
    return resp


# list of items that may have an other option
OTHER_LIST = [
    "mood_diagnoses",
    "mood_treatment",
    "anxiety_diagnoses",
    "anxiety_treatment",
    "attention_diagnoses",
    "attention_treatment",
]


def extract_responses(
    responses: dict[str, str | None | dict[str, str | int]],
    decoders: dict[str, Callable],
) -> dict[str, Any]:
    """Process a dictionary of survey responeses with the dictionary of decoders and extract transformed respones
    Parameters
    ----------
        responeses : dict
            Dictionary of survey responese
        decodders : dict
            Dictionary with same keys as responese with transformer functions for each field
    Returns
    -------
        results : dict
            Dictionary of results

    """
    result = {}
    for k, v in responses.items():
        decoder = decoders.get(k)
        if decoder is not None:
            result.update(decoder(k, v))
    for field in OTHER_LIST:
        if result[f"{field}__other"]:
            result[f"{field}__otherresp"]: responses[f"{field}-Comment"]
    return result


# definition of transformers for each field in the cogmood survey
SURVEY_DECODE = {
    "race": ohe_fac(
        [
            "American Indian/Alaska Native",
            "Asian",
            "Native Hawaiian or Other Pacific Islander",
            "Black or African American",
            "White",
        ]
    ),
    "ethnicity": ohe_fac(
        ["Not Hispanic or Latino", "Hispanic or Latino"], force_list=True
    ),
    "sex_at_birth": ohe_fac(
        ["Female", "Male", "Prefer not to answer"], force_list=True
    ),
    "age": num_code,
    "ladder_resp": str_code,
    "ongoing_mentalhealth": yn_code,
    "mentalhealth_daily_impact": yn_code,
    "experience_depression": yn_code,
    "experience_anxiety": yn_code,
    "have_adhd": yn_code,
    "attn__1": yn_code,
    "fried": str_code,
    "mood_pro_diagnosis": yn_code,
    "mood_diagnoses": ohe_fac(
        [
            "Major Depressive Disorder",
            "Persistent Depressive Disorder",
            "(Dysthymia)Premenstrual Dysphoric Disorder",
            "Bipolar I Disorder",
            "Bipolar II Disorder",
            "Cyclothymic Disorder",
        ],
        other=True,
        none=True,
    ),
    "mood_first_dx_years": num_code,
    "mood_treatment": ohe_fac(
        [
            "I am not receiving treatment for a mood disorder",
            "One-on-one talk therapy with a professional",
            "Group therapy",
            "Selective Serotonin Reuptake Inhibitor (SSRI), such as: Prozac (fluoxotine), Celexa (citalopram), Lexapro (escitalopram), Paroxetine (Paxil), Sertraline (Zoloft)",
            "Serotonin and Norepinephrine Reuptake Inhibitor (SNRI), such as: Cymbalta (duloxetine), Effexor (venalfaxine)",
            "Tricyclic Antidepressant (TCA), such as: Anafranil (clomipramine), Sinequan (doxepin)",
            "Monoamine Oxidase Inhibitor (MAOI), such as: Emsam (selegiline), Marplan (isocarboxaxid)",
            "Serotonin Antagonist and Reuptake Inhibitors, such as: Oleptro (trazodone), Brintellix (vortioxetine)",
            "Remeron (mirtazapine)",
            "Symbax (olanzapine/fluoxotine)",
            "Wellbutrin (buproprion)",
            "Lithium",
            "Depakane (valproate), Epival (divalproex)",
            "Tegretol (carbamazepine), Trileptal (oxcarbazepine)",
            "Lamictal (lamotrigine)",
            "Haloperidol (haldol decanoate)",
            "Abilify (aripiprazole), Saphris (asenapine), Vraylar (cariprazine), Zyprexa (olanzapine), Risperdal (risperdone)",
            "Latuda (lurasidone)",
            "Caplyta (lumaterperone), Seroquel (quetiapine)",
        ],
        other=True,
        none=True,
    ),
    "mood_med_today": yn_code,
    "mood_bothered": yn_code,
    "mood_bothered_today": yn_code,
    "anxiety_pro_diagnosis": yn_code,
    "anxiety_diagnoses": ohe_fac(
        [
            "Generalized Anxiety Disorder",
            "Separation Anxiety Disorder",
            "Agoraphobia",
            "Specific Phobia",
            "Social Anxiety Disorder (Social Phobia)",
            "Panic Disorder",
            "Panic Attack",
            "Selective Mutism",
        ],
        other=True,
        none=True,
    ),
    "anxiety_first_dx_years": num_code,
    "anxiety_treatment": ohe_fac(
        [
            "I am not receiving treatment for an anxiety disorder",
            "One-on-one talk therapy with a professional",
            "Group therapy",
            "Selective Serotonin Reuptake Inhibitor (SSRI), such as: Prozac (fluozotine), Celexa (citalopram), Lexapro (escitalopram), Paroxetine (Paxil), Sertraline (Zoloft)",
            "Serotonin and Norepinephrine Reuptake Inhibitor (SNRI), such as: Cymbalta (duloxetine), Effexor (venalfaxine)",
            "Tricyclic Antidepressant (TCA), such as: Anafranil (clomipramine), Sinequan (doxepin)",
            "Benzodiazepine, such as: Xanax (alprazolam), Valium (diazepam), Ativan (lorazepam), Librium (chlordiazepoxide)",
            "Monoamine Oxidase Inhibitor (MAOI), such as: Emsam (selegiline), Marplan (isocarboxaxid)",
            "Beta-blocker, such as: Tenormin (atenolol), Inderal (propranolol)",
            "BuSpar (buspirone)",
        ],
        other=True,
        none=True,
    ),
    "anxiety_med_today": yn_code,
    "anxiety_bothered": yn_code,
    "anxiety_bothered_today": yn_code,
    "attention_pro_diagnosis": yn_code,
    "attention_diagnoses": ohe_fac(
        [
            "Attention-Deficit/Hyperactivity Disorder (ADHD)",
            "Attention-Deficit Disorder (ADD)",
        ],
        other=True,
        none=True,
    ),
    "attention_first_dx_years": num_code,
    "attention_treatment": ohe_fac(
        [
            "I am not receiving treatment for an attention disorder",
            "One-on-one talk therapy with a professional",
            "Group therapy",
            "Amphetamine, such as Dexedrine (dextroamphetamine), Adderall (amphetamine salts), Vyvanse (lisdexamfetamine)",
            "Methylphenidate, such as Concerta, Ritalin, Focalin (dexmethylphenidate)",
            "Strattera (atomoxetine)",
            "Wellbutrin (buproprion)",
            "Intuniv (guanfacine)",
            "Catapres, Kapvay (clonidine)",
        ],
        other=True,
        none=True,
    ),
    "anxiety_med_today": yn_code,
    "anxiety_bothered": yn_code,
    "anxiety_bothered_today": yn_code,
    "baaars_inattention": survey_matrix,
    "baaars_hyperactivity": survey_matrix,
    "baaars_impulsivity": survey_matrix,
    "baaars_sct": survey_matrix,
    "gad7_": survey_matrix,
    "phq8_": survey_matrix,
    "hitop_01": survey_matrix,
    "hitop_02": survey_matrix,
    "hitop_03": survey_matrix,
    "hitop_04": survey_matrix,
    "hitop_05": survey_matrix,
    "hitop_06": survey_matrix,
    "hitop_07": survey_matrix,
    "hitop_08": survey_matrix,
    "hitop_09": survey_matrix,
    "hitop_10": survey_matrix,
    "hitop_11": survey_matrix,
    "hitop_12": survey_matrix,
    "hitop_13": survey_matrix,
    "hitop_14": survey_matrix,
    "hitop_15": survey_matrix,
    "today_1": ohe_fac(
        [
            "todaybaars_inattentive_1",
            "todaybaars_inattentive_2",
            "todaybaars_inattentive_3",
            "todaybaars_inattentive_4",
            "todaybaars_inattentive_5",
            "todaybaars_inattentive_6",
            "todaybaars_inattentive_7",
            "todaybaars_inattentive_8",
            "todaybaars_inattentive_9",
            "todaybaars_hyperactivity_1",
            "todaybaars_hyperactivity_2",
            "todaybaars_hyperactivity_3",
            "todaybaars_hyperactivity_4",
            "todaybaars_hyperactivity_5",
            "todaybaars_impulsivity_1",
            "todayattn__1",
            "todaybaars_impulsivity_2",
            "todaybaars_impulsivity_3",
            "todaybaars_impulsivity_4",
            "todaybaars_sct_1",
            "todaybaars_sct_2",
            "todaybaars_sct_3",
            "todaybaars_sct_4",
            "todaybaars_sct_5",
            "todaybaars_sct_6",
            "todaybaars_sct_7",
            "todaybaars_sct_8",
            "todaybaars_sct_9",
            "today1__none",
        ]
    ),
    "today_2": ohe_fac(
        [
            "todaygad7__1",
            "todaygad7__2",
            "todaygad7__3",
            "todaygad7__4",
            "todaygad7__5",
            "todaygad7__6",
            "todaygad7__7",
            "todayphq8__1",
            "todayphq8__2",
            "todayphq8__3",
            "todayphq8__4",
            "todayphq8__5",
            "todayphq8__6",
            "todayphq8__7",
            "todayphq8__8",
            "today2__none",
        ]
    ),
    "today_3": ohe_fac(
        [
            "todayhitop_anhdep_1",
            "todayhitop_sepinsec_1",
            "todayhitop_anxwor_1",
            "todayhitop_welbe_1",
            "todayhitop_appgn_1",
            "todayhitop_anhdep_2",
            "todayhitop_sepinsec_2",
            "todayhitop_anxwor_2",
            "todayhitop_sepinsec_3",
            "todayhitop_socanx_1",
            "todayhitop_anxwor_3",
            "todayhitop_socanx_2",
            "todayhitop_hypsom_1",
            "todayhitop_anhdep_3",
            "todayhitop_cogprb_1",
            "todayhitop_welbe_2",
            "todayhitop_welbe_3",
            "todayattn__2",
            "todayhitop_appgn_2",
            "todayhitop_sepinsec_4",
            "todayhitop_socanx_3",
            "todayhitop_indec_1",
            "todayhitop_socanx_4",
            "todayhitop_appls_1",
            "todayhitop_anhdep_4",
            "todayhitop_appgn_3",
            "todayhitop_anhdep_5",
            "todayhitop_anhdep_6",
            "todayhitop_socanx_5",
            "today3__none",
        ]
    ),
    "today_4": ohe_fac(
        [
            "todayhitop_anhdep_7",
            "todayhitop_insom_1",
            "todayhitop_panic_1",
            "todayhitop_indec_2",
            "todayhitop_welbe_4",
            "todayhitop_appls_2",
            "todayhitop_sitphb_1",
            "todayhitop_anxwor_4",
            "todayhitop_cogprb_2",
            "todayhitop_socanx_6",
            "todayhitop_sepinsec_5",
            "todayhitop_anhdep_8",
            "todayhitop_anxwor_5",
            "todayhitop_panic_2",
            "todayhitop_socanx_7",
            "todayhitop_socanx_8",
            "todayhitop_welbe_5",
            "todayhitop_welbe_6",
            "todayhitop_insom_2",
            "todayhitop_welbe_7",
            "todayhitop_sitphb_2",
            "todayhitop_welbe_8",
            "todayhitop_welbe_9",
            "todayhitop_cogprb_3",
            "todayhitop_anhdep_9",
            "todayhitop_welbe_10",
            "todayhitop_sitphb_3",
            "todayhitop_hypsom_2",
            "todayhitop_hypsom_3",
            "today4__none",
        ]
    ),
    "today_5": ohe_fac(
        [
            "todayhitop_anxwor_6",
            "todayhitop_socanx_9",
            "todayhitop_indec_3",
            "todayhitop_sepinsec_6",
            "todayhitop_sitphb_4",
            "todayhitop_anxwor_7",
            "todayhitop_cogprb_4",
            "todayhitop_anhdep_10",
            "todayhitop_appgn_4",
            "todayhitop_insom_3",
            "todayhitop_sitphb_5",
            "todayhitop_shmglt_1",
            "todayhitop_sepinsec_7",
            "todayhitop_hypsom_4",
            "todayhitop_panic_3",
            "todayhitop_socanx_10",
            "todayhitop_panic_4",
            "todayhitop_hypsom_5",
            "todayhitop_insom_4",
            "todayhitop_shmglt_2",
            "todayhitop_panic_5",
            "todayhitop_shmglt_3",
            "todayhitop_shmglt_4",
            "todayattn__3",
            "todayhitop_appls_3",
            "todayhitop_sepinsec_8",
            "todayhitop_panic_6",
            "today5__none",
        ]
    ),
    "handedness": ohe_fac(
        ["Right Handed", "Ambidextrous", "Left Handed"], force_list=True
    ),
    "fatigue": likert_code,
    "meal_type": ohe_fac(["More", "Less", "About the same"], force_list=True),
    "hunger": likert_code,
}
