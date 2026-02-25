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
    "attention_med_today": yn_code,
    "attention_bothered": yn_code,
    "attention_bothered_today": yn_code,
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


COL_LUT = {
    'mood_diagnoses__major_depressive_disorder': 'mood_diagnoses__mdd',
    'mood_diagnoses__persistent_depressive_disorder': 'mood_diagnoses__pdd',
    'mood_diagnoses___dysthymia_premenstrual_dysphoric_disorder': 'mood_diagnoses__premenstrual',
    'mood_diagnoses__bipolar_i_disorder': 'mood_diagnoses__bp1',
    'mood_diagnoses__bipolar_ii_disorder': 'mood_diagnoses__bp2',
    'mood_diagnoses__cyclothymic_disorder': 'mood_diagnoses__ct',
    'mood_diagnoses__none': 'mood_diagnoses__pna',
    'mood_treatment__i_am_not_receiving_treatment_for_a_mood_disorder': 'mood_treatment__none',
    'mood_treatment__one_on_one_talk_therapy_with_a_professional': 'mood_treatment__1on1',
    'mood_treatment__group_therapy': 'mood_treatment__group',
    'mood_treatment__selective_serotonin_reuptake_inhibitor_ssri_such_as_prozac_fluoxotine_celexa_citalopram_lexapro_escitalopram_paroxetine_paxil_sertraline_zoloft_': 'mood_treatment__ssri',
    'mood_treatment__serotonin_and_norepinephrine_reuptake_inhibitor_snri_such_as_cymbalta_duloxetine_effexor_venalfaxine_': 'mood_treatment__snri',
    'mood_treatment__tricyclic_antidepressant_tca_such_as_anafranil_clomipramine_sinequan_doxepin_': 'mood_treatment__tricylic',
    'mood_treatment__monoamine_oxidase_inhibitor_maoi_such_as_emsam_selegiline_marplan_isocarboxaxid_': 'mood_treatment__maoi',
    'mood_treatment__serotonin_antagonist_and_reuptake_inhibitors_such_as_oleptro_trazodone_brintellix_vortioxetine_': 'mood_treatment__sari',
    'mood_treatment__remeron_mirtazapine_': 'mood_treatment__tetracyclic',
    'mood_treatment__symbax_olanzapine_fluoxotine_': 'mood_treatment__olanzapine_fluoxotine',
    'mood_treatment__wellbutrin_buproprion_': 'mood_treatment__buproprion',
    'mood_treatment__lithium': 'mood_treatment__li',
    'mood_treatment__depakane_valproate_epival_divalproex_': 'mood_treatment__valproate',
    'mood_treatment__tegretol_carbamazepine_trileptal_oxcarbazepine_': 'mood_treatment__carbazepine',
    'mood_treatment__lamictal_lamotrigine_': 'mood_treatment__lamotrigine',
    'mood_treatment__haloperidol_haldol_decanoate_': 'mood_treatment__haldol',
    'mood_treatment__abilify_aripiprazole_saphris_asenapine_vraylar_cariprazine_zyprexa_olanzapine_risperdal_risperdone_': 'mood_treatment__atypical_antipsychotic',
    'mood_treatment__latuda_lurasidone_': 'mood_treatment__lurasidone',
    'mood_treatment__caplyta_lumaterperone_seroquel_quetiapine_': 'mood_treatment__lumaterperone',
    'mood_treatment__none': 'mood_treatment__pna',
    'anxiety_diagnoses__generalized_anxiety_disorder': 'anxiety_diagnoses__gad',
    'anxiety_diagnoses__separation_anxiety_disorder': 'anxiety_diagnoses__sepad',
    'anxiety_diagnoses__social_anxiety_disorder_social_phobia_': 'anxiety_diagnoses__socad',
    'anxiety_diagnoses__none': 'anxiety_diagnoses__pna',
    'anxiety_treatment__i_am_not_receiving_treatment_for_an_anxiety_disorder': 'anxiety_treatment__none',
    'anxiety_treatment__one_on_one_talk_therapy_with_a_professional': 'anxiety_treatment__1on1',
    'anxiety_treatment__group_therapy': 'anxiety_treatment__group',
    'anxiety_treatment__selective_serotonin_reuptake_inhibitor_ssri_such_as_prozac_fluozotine_celexa_citalopram_lexapro_escitalopram_paroxetine_paxil_sertraline_zoloft_': 'anxiety_treatment__ssri',
    'anxiety_treatment__serotonin_and_norepinephrine_reuptake_inhibitor_snri_such_as_cymbalta_duloxetine_effexor_venalfaxine_': 'anxiety_treatment__snri',
    'anxiety_treatment__tricyclic_antidepressant_tca_such_as_anafranil_clomipramine_sinequan_doxepin_': 'anxiety_treatment__tricyclic',
    'anxiety_treatment__benzodiazepine_such_as_xanax_alprazolam_valium_diazepam_ativan_lorazepam_librium_chlordiazepoxide_': 'anxiety_treatment__benzodiazepine',
    'anxiety_treatment__monoamine_oxidase_inhibitor_maoi_such_as_emsam_selegiline_marplan_isocarboxaxid_': 'anxiety_treatment__maoi',
    'anxiety_treatment__beta_blocker_such_as_tenormin_atenolol_inderal_propranolol_': 'anxiety_treatment__beta_blocker',
    'anxiety_treatment__buspar_buspirone_': 'anxiety_treatment__buspirone',
    'anxiety_treatment__none': 'anxiety_treatment__pna',
    'attention_diagnoses__attention_deficit_hyperactivity_disorder_adhd_': 'attention_diagnoses__adhd',
    'attention_diagnoses__attention_deficit_disorder_add_': 'attention_diagnoses__add',
    'attention_diagnoses__none': 'attention_diagnoses__pna',
    'attention_treatment__i_am_not_receiving_treatment_for_an_attention_disorder': 'attention_treatment__none',
    'attention_treatment__one_on_one_talk_therapy_with_a_professional': 'attention_treatment__1on1',
    'attention_treatment__group_therapy': 'attention_treatment__group',
    'attention_treatment__amphetamine_such_as_dexedrine_dextroamphetamine_adderall_amphetamine_salts_vyvanse_lisdexamfetamine_': 'attention_treatment__amphetamine',
    'attention_treatment__methylphenidate_such_as_concerta_ritalin_focalin_dexmethylphenidate_': 'attention_treatment__methylphenidate',
    'attention_treatment__strattera_atomoxetine_': 'attention_treatment__stomoxetine',
    'attention_treatment__wellbutrin_buproprion_': 'attention_treatment__buproprion',
    'attention_treatment__intuniv_guanfacine_': 'attention_treatment__guanfacine',
    'attention_treatment__catapres_kapvay_clonidine_': 'attention_treatment__clonidine',
    'attention_treatment__none': 'attention_treatment__pna',
    'today_1__todaybaars_inattentive_1': 'todaybaars_inattentive_1',
     'today_1__todaybaars_inattentive_2': 'todaybaars_inattentive_2',
     'today_1__todaybaars_inattentive_3': 'todaybaars_inattentive_3',
     'today_1__todaybaars_inattentive_4': 'todaybaars_inattentive_4',
     'today_1__todaybaars_inattentive_5': 'todaybaars_inattentive_5',
     'today_1__todaybaars_inattentive_6': 'todaybaars_inattentive_6',
     'today_1__todaybaars_inattentive_7': 'todaybaars_inattentive_7',
     'today_1__todaybaars_inattentive_8': 'todaybaars_inattentive_8',
     'today_1__todaybaars_inattentive_9': 'todaybaars_inattentive_9',
     'today_1__todaybaars_hyperactivity_1': 'todaybaars_hyperactivity_1',
     'today_1__todaybaars_hyperactivity_2': 'todaybaars_hyperactivity_2',
     'today_1__todaybaars_hyperactivity_3': 'todaybaars_hyperactivity_3',
     'today_1__todaybaars_hyperactivity_4': 'todaybaars_hyperactivity_4',
     'today_1__todaybaars_hyperactivity_5': 'todaybaars_hyperactivity_5',
     'today_1__todaybaars_impulsivity_1': 'todaybaars_impulsivity_1',
     'today_1__todayattn_1': 'todayattn_1',
     'today_1__todaybaars_impulsivity_2': 'todaybaars_impulsivity_2',
     'today_1__todaybaars_impulsivity_3': 'todaybaars_impulsivity_3',
     'today_1__todaybaars_impulsivity_4': 'todaybaars_impulsivity_4',
     'today_1__todaybaars_sct_1': 'todaybaars_sct_1',
     'today_1__todaybaars_sct_2': 'todaybaars_sct_2',
     'today_1__todaybaars_sct_3': 'todaybaars_sct_3',
     'today_1__todaybaars_sct_4': 'todaybaars_sct_4',
     'today_1__todaybaars_sct_5': 'todaybaars_sct_5',
     'today_1__todaybaars_sct_6': 'todaybaars_sct_6',
     'today_1__todaybaars_sct_7': 'todaybaars_sct_7',
     'today_1__todaybaars_sct_8': 'todaybaars_sct_8',
     'today_1__todaybaars_sct_9': 'todaybaars_sct_9',
     'today_1__today1_none': 'today1_none',
     'today_2__todaygad7_1': 'todaygad7_1',
     'today_2__todaygad7_2': 'todaygad7_2',
     'today_2__todaygad7_3': 'todaygad7_3',
     'today_2__todaygad7_4': 'todaygad7_4',
     'today_2__todaygad7_5': 'todaygad7_5',
     'today_2__todaygad7_6': 'todaygad7_6',
     'today_2__todaygad7_7': 'todaygad7_7',
     'today_2__todayphq8_1': 'todayphq8_1',
     'today_2__todayphq8_2': 'todayphq8_2',
     'today_2__todayphq8_3': 'todayphq8_3',
     'today_2__todayphq8_4': 'todayphq8_4',
     'today_2__todayphq8_5': 'todayphq8_5',
     'today_2__todayphq8_6': 'todayphq8_6',
     'today_2__todayphq8_7': 'todayphq8_7',
     'today_2__todayphq8_8': 'todayphq8_8',
     'today_2__today2_none': 'today2_none',
     'today_3__todayhitop_anhdep_1': 'todayhitop_anhdep_1',
     'today_3__todayhitop_sepinsec_1': 'todayhitop_sepinsec_1',
     'today_3__todayhitop_anxwor_1': 'todayhitop_anxwor_1',
     'today_3__todayhitop_welbe_1': 'todayhitop_welbe_1',
     'today_3__todayhitop_appgn_1': 'todayhitop_appgn_1',
     'today_3__todayhitop_anhdep_2': 'todayhitop_anhdep_2',
     'today_3__todayhitop_sepinsec_2': 'todayhitop_sepinsec_2',
     'today_3__todayhitop_anxwor_2': 'todayhitop_anxwor_2',
     'today_3__todayhitop_sepinsec_3': 'todayhitop_sepinsec_3',
     'today_3__todayhitop_socanx_1': 'todayhitop_socanx_1',
     'today_3__todayhitop_anxwor_3': 'todayhitop_anxwor_3',
     'today_3__todayhitop_socanx_2': 'todayhitop_socanx_2',
     'today_3__todayhitop_hypsom_1': 'todayhitop_hypsom_1',
     'today_3__todayhitop_anhdep_3': 'todayhitop_anhdep_3',
     'today_3__todayhitop_cogprb_1': 'todayhitop_cogprb_1',
     'today_3__todayhitop_welbe_2': 'todayhitop_welbe_2',
     'today_3__todayhitop_welbe_3': 'todayhitop_welbe_3',
     'today_3__todayattn_2': 'todayattn_2',
     'today_3__todayhitop_appgn_2': 'todayhitop_appgn_2',
     'today_3__todayhitop_sepinsec_4': 'todayhitop_sepinsec_4',
     'today_3__todayhitop_socanx_3': 'todayhitop_socanx_3',
     'today_3__todayhitop_indec_1': 'todayhitop_indec_1',
     'today_3__todayhitop_socanx_4': 'todayhitop_socanx_4',
     'today_3__todayhitop_appls_1': 'todayhitop_appls_1',
     'today_3__todayhitop_anhdep_4': 'todayhitop_anhdep_4',
     'today_3__todayhitop_appgn_3': 'todayhitop_appgn_3',
     'today_3__todayhitop_anhdep_5': 'todayhitop_anhdep_5',
     'today_3__todayhitop_anhdep_6': 'todayhitop_anhdep_6',
     'today_3__todayhitop_socanx_5': 'todayhitop_socanx_5',
     'today_3__today3_none': 'today3_none',
     'today_4__todayhitop_anhdep_7': 'todayhitop_anhdep_7',
     'today_4__todayhitop_insom_1': 'todayhitop_insom_1',
     'today_4__todayhitop_panic_1': 'todayhitop_panic_1',
     'today_4__todayhitop_indec_2': 'todayhitop_indec_2',
     'today_4__todayhitop_welbe_4': 'todayhitop_welbe_4',
     'today_4__todayhitop_appls_2': 'todayhitop_appls_2',
     'today_4__todayhitop_sitphb_1': 'todayhitop_sitphb_1',
     'today_4__todayhitop_anxwor_4': 'todayhitop_anxwor_4',
     'today_4__todayhitop_cogprb_2': 'todayhitop_cogprb_2',
     'today_4__todayhitop_socanx_6': 'todayhitop_socanx_6',
     'today_4__todayhitop_sepinsec_5': 'todayhitop_sepinsec_5',
     'today_4__todayhitop_anhdep_8': 'todayhitop_anhdep_8',
     'today_4__todayhitop_anxwor_5': 'todayhitop_anxwor_5',
     'today_4__todayhitop_panic_2': 'todayhitop_panic_2',
     'today_4__todayhitop_socanx_7': 'todayhitop_socanx_7',
     'today_4__todayhitop_socanx_8': 'todayhitop_socanx_8',
     'today_4__todayhitop_welbe_5': 'todayhitop_welbe_5',
     'today_4__todayhitop_welbe_6': 'todayhitop_welbe_6',
     'today_4__todayhitop_insom_2': 'todayhitop_insom_2',
     'today_4__todayhitop_welbe_7': 'todayhitop_welbe_7',
     'today_4__todayhitop_sitphb_2': 'todayhitop_sitphb_2',
     'today_4__todayhitop_welbe_8': 'todayhitop_welbe_8',
     'today_4__todayhitop_welbe_9': 'todayhitop_welbe_9',
     'today_4__todayhitop_cogprb_3': 'todayhitop_cogprb_3',
     'today_4__todayhitop_anhdep_9': 'todayhitop_anhdep_9',
     'today_4__todayhitop_welbe_10': 'todayhitop_welbe_10',
     'today_4__todayhitop_sitphb_3': 'todayhitop_sitphb_3',
     'today_4__todayhitop_hypsom_2': 'todayhitop_hypsom_2',
     'today_4__todayhitop_hypsom_3': 'todayhitop_hypsom_3',
     'today_4__today4_none': 'today4_none',
     'today_5__todayhitop_anxwor_6': 'todayhitop_anxwor_6',
     'today_5__todayhitop_socanx_9': 'todayhitop_socanx_9',
     'today_5__todayhitop_indec_3': 'todayhitop_indec_3',
     'today_5__todayhitop_sepinsec_6': 'todayhitop_sepinsec_6',
     'today_5__todayhitop_sitphb_4': 'todayhitop_sitphb_4',
     'today_5__todayhitop_anxwor_7': 'todayhitop_anxwor_7',
     'today_5__todayhitop_cogprb_4': 'todayhitop_cogprb_4',
     'today_5__todayhitop_anhdep_10': 'todayhitop_anhdep_10',
     'today_5__todayhitop_appgn_4': 'todayhitop_appgn_4',
     'today_5__todayhitop_insom_3': 'todayhitop_insom_3',
     'today_5__todayhitop_sitphb_5': 'todayhitop_sitphb_5',
     'today_5__todayhitop_shmglt_1': 'todayhitop_shmglt_1',
     'today_5__todayhitop_sepinsec_7': 'todayhitop_sepinsec_7',
     'today_5__todayhitop_hypsom_4': 'todayhitop_hypsom_4',
     'today_5__todayhitop_panic_3': 'todayhitop_panic_3',
     'today_5__todayhitop_socanx_10': 'todayhitop_socanx_10',
     'today_5__todayhitop_panic_4': 'todayhitop_panic_4',
     'today_5__todayhitop_hypsom_5': 'todayhitop_hypsom_5',
     'today_5__todayhitop_insom_4': 'todayhitop_insom_4',
     'today_5__todayhitop_shmglt_2': 'todayhitop_shmglt_2',
     'today_5__todayhitop_panic_5': 'todayhitop_panic_5',
     'today_5__todayhitop_shmglt_3': 'todayhitop_shmglt_3',
     'today_5__todayhitop_shmglt_4': 'todayhitop_shmglt_4',
     'today_5__todayattn_3': 'todayattn_3',
     'today_5__todayhitop_appls_3': 'todayhitop_appls_3',
     'today_5__todayhitop_sepinsec_8': 'todayhitop_sepinsec_8',
     'today_5__todayhitop_panic_6': 'todayhitop_panic_6',
     'today_5__today5_none': 'today5_none'
}


SCALES = [
    ('baars', None),
    ('phq8', None),
    ('gad7', None),
    ('hitop', None),
    ('baars', 'hyperactivity'),
    ('baars', 'impulsivity'),
    ('baars', 'inattentive'),
    ('baars', 'sct'),
    ('hitop', 'anhdep'),
    ('hitop', 'sepinsec'),
    ('hitop', 'anxwor'),
    ('hitop', 'welbe'),
    ('hitop', 'appgn'),
    ('hitop', 'socanx'),
    ('hitop', 'hypsom'),
    ('hitop', 'cogprb'),
    ('hitop', 'indec'),
    ('hitop', 'appls'),
    ('hitop', 'insom'),
    ('hitop', 'shmglt'),
    ('attnbin', None),
]
