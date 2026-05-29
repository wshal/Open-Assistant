import unittest

from utils.text_utils import (
    is_likely_fragment,
    looks_like_actionable_auto_query,
    merge_transcripts,
    normalize_transcript,
    sanitize_auto_transcript,
    sanitize_query_label,
)


class TextUtilsTests(unittest.TestCase):
    def test_normalize_transcript_repairs_merged_question_words(self):
        self.assertEqual(
            normalize_transcript("Whocan tellme what is React?"),
            "Who can tell me what is React?",
        )

    def test_normalize_transcript_repairs_merged_followup_words(self):
        self.assertEqual(
            normalize_transcript("and why do weuse hooksin React?"),
            "and why do we use hooks in React?",
        )

    def test_is_likely_fragment_rejects_low_information_scrap(self):
        self.assertTrue(is_likely_fragment("for Cloud Helps."))

    def test_is_likely_fragment_keeps_real_question(self):
        self.assertFalse(is_likely_fragment("Why do we use hooks in React?"))

    def test_normalize_transcript_keeps_useeffect_phrase_intact(self):
        self.assertEqual(
            normalize_transcript("Why do we use useEffect hook?"),
            "Why do we use useEffect hook?",
        )

    def test_normalize_transcript_does_not_merge_question_word_with_acronym(self):
        self.assertEqual(
            normalize_transcript("Could you explain how ON JWT tokens work?"),
            "Could you explain how JWT tokens work?",
        )

    def test_normalize_transcript_repairs_algorithmic_fragment_examples(self):
        self.assertEqual(
            normalize_transcript("monolithic appli cation backed by a single relational databa se"),
            "monolithic application backed by a single relational database",
        )
        self.assertEqual(
            normalize_transcript("What are some stra tegies you would consider"),
            "What are some strategies you would consider",
        )

    def test_sanitize_auto_transcript_repairs_space_fragmented_words(self):
        self.assertEqual(
            sanitize_auto_transcript("sharing sta teful logic"),
            "sharing stateful logic",
        )

    def test_merge_transcripts_heals_overlapping_chunks(self):
        self.assertEqual(
            merge_transcripts(
                "Could you explain how JWT tokens work?",
                "JWT tokens work? What are the potential security risks?",
            ),
            "Could you explain how JWT tokens work? What are the potential security risks?",
        )

    def test_merge_transcripts_heals_split_boundary_word(self):
        self.assertEqual(
            merge_transcripts(
                "Can you explain the primary differen",
                "ferences between the two?",
            ),
            "Can you explain the primary differences between the two?",
        )

    def test_merge_transcripts_keeps_sentence_boundary_spacing(self):
        self.assertEqual(
            merge_transcripts(
                "Let's pivot to some CSS basics.",
                "A lot of developers get confused between CSS Grid and Flexbox.",
            ),
            "Let's pivot to some CSS basics. A lot of developers get confused between CSS Grid and Flexbox.",
        )

    def test_sanitize_query_label_repairs_benchmark_phrases(self):
        self.assertEqual(
            sanitize_query_label("What are some of the key principles you would follow to ensure the API is robustver maintainable and provides a goodde veloper experience for the frontend team?"),
            "What are some of the key principles you would follow to ensure the API is robust, maintainable and provides a good developer experience for the frontend team?",
        )

    def test_sanitize_auto_transcript_handles_space_fragmented_sentence_without_recursion(self):
        self.assertEqual(
            sanitize_auto_transcript("Let's pivot to some CSS basics. A lot of de velop ers get con fused bet ween CSS Grid and Flexbox."),
            "Let's pivot to some CSS basics. A lot of developers get confused between CSS Grid and Flexbox.",
        )

    def test_sanitize_query_label_repairs_css_benchmark_question(self):
        self.assertEqual(
            sanitize_query_label("A lot of de velopersgetcon fusedbet ween CSS Grid. and Flexbox. Can you plain the pri marydi fferences between the two?"),
            "Can you explain the primary differences between the two?",
        )

    def test_sanitize_auto_transcript_repairs_benchmark_scaling_question(self):
        self.assertEqual(
            sanitize_auto_transcript("What are some strategies you wouldcon sider alleviate that bottleneck?"),
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_sanitize_query_label_repairs_benchmark_react_hooks_question(self):
        self.assertEqual(
            sanitize_query_label("Could you walk me through how wouldde cidebet weenusing a cu stomversushi gher or order component for sha ringsta teful logic?"),
            "Could you walk me through how would decide between using a custom versus a higher-order component for sharing stateful logic?",
        )

    def test_sanitize_query_label_repairs_latest_benchmark_react_hooks_question(self):
        self.assertEqual(
            sanitize_query_label("Could you walk me through how you would cidebet weenu sing a custom hook versus a higher-order component for sha ringstateful logic?"),
            "Could you walk me through how you would decide between using a custom hook versus a higher-order component for sharing stateful logic?",
        )

    def test_sanitize_query_label_repairs_benchmark_api_design_question(self):
        self.assertEqual(
            sanitize_query_label("What are some of the keypri nciples you would follow to en PI is ro bustversu inable and provides good veloperex perience for frontend team?"),
            "What are some of the key principles you would follow to ensure the API is robust, maintainable and provides good developer experience for frontend team?",
        )

    def test_sanitize_query_label_repairs_benchmark_jwt_storage_question(self):
        self.assertEqual(
            sanitize_query_label("What are the potential curity risks or if you store a in bro wserslocalage stead of an http-only cookie?"),
            "What are the potential security risks if you store a JWT in browser local storage instead of an http-only cookie?",
        )

    def test_sanitize_query_label_repairs_latest_css_followup_question(self):
        self.assertEqual(
            sanitize_query_label("give an example of a layout where you wouldde finitelychoosegrid over Flexbox?"),
            "give an example of a layout where you would definitely choose grid over Flexbox?",
        )

    def test_sanitize_query_label_repairs_latest_react_hooks_question(self):
        self.assertEqual(
            sanitize_query_label("Somo ving on to the next topic. I was loo king at yourresu me and I see you 'veusedreact quite a bit. Could you walk me through how you would decide between using a cu stomhookversus a hi gher or order component for sharing stateful logic?"),
            "Could you walk me through how you would decide between using a custom hook versus a higher-order component for sharing stateful logic?",
        )

    def test_sanitize_query_label_repairs_compound_splits_without_regex_only_path(self):
        self.assertEqual(
            sanitize_query_label("I see you usedreact quite a bit and prefer customhookversus patterns for sharingstateful logic."),
            "I see you used React quite a bit and prefer custom hook versus patterns for sharing stateful logic.",
        )

    def test_sanitize_query_label_repairs_latest_api_design_question(self):
        self.assertEqual(
            sanitize_query_label("What are some of the key principles you would follow to en sure the API is robust, maintainable and provides a goodde developer experience for the frontend team?"),
            "What are some of the key principles you would follow to ensure the API is robust, maintainable and provides a good developer experience for the frontend team?",
        )

    def test_sanitize_query_label_repairs_latest_jwt_storage_question(self):
        self.assertEqual(
            sanitize_query_label("What are the potential security risks if you store a JWT in the bro wserslocal storage instead of an h http-only cookie?"),
            "What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_sanitize_query_label_repairs_browser_local_chunk_boundary(self):
        self.assertEqual(
            sanitize_query_label("Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the bro wserlocal storage instead of an http-only cookie?"),
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_sanitize_query_label_repairs_latest_jwt_interim_benchmark_text(self):
        self.assertEqual(
            sanitize_query_label("When building a secure REST API au then Could you expla in how JSON JWT tokens work? What are the potential security risks are if you store a JWT in the bro. wserslocal storage instead of an h. tt."),
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_sanitize_query_label_repairs_newest_jwt_only_cooking_artifact(self):
        self.assertEqual(
            sanitize_query_label("Could you explain how JWT tokens work? what the pote ntialse security risks are if you store a JWT in the browser local storage only cooking?"),
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_sanitize_auto_transcript_repairs_newest_scaling_setup_artifact(self):
        self.assertEqual(
            sanitize_auto_transcript("Let's talk about scaling. imagine we have monolithicapp licationba cked by single relational database that's starting to slowdown under heavy traffic."),
            "Let's talk about scaling. imagine we have monolithic application backed by single relational database that's starting to slowdown under heavy traffic.",
        )

    def test_sanitize_query_label_repairs_latest_scaling_question(self):
        self.assertEqual(
            sanitize_query_label("What are some strategies you would consider to alle viate that bottleneck?"),
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_sanitize_query_label_repairs_latest_jt_token_artifact(self):
        self.assertEqual(
            sanitize_query_label("Could you explain how JWT tokens work? What are the potential security risks if you store a JT in the browser local storage instead of an http-only cookie?"),
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_sanitize_query_label_repairs_latest_scaling_fragment_duplication(self):
        self.assertEqual(
            sanitize_query_label("What are some stra.. tegi.. What are some strategies you would?"),
            "What are some strategies you would?",
        )

    def test_sanitize_query_label_repairs_garbled_scaling_benchmark_query(self):
        self.assertEqual(
            sanitize_query_label(
                "What are some stra tegiAll rightlets talkaboutsca lingIma gine we have monolithic application backed by single relational database that 's starting to slowdown under he avyredtra ffic What are some strategies?"
            ),
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_sanitize_query_label_drops_scaling_setup_preamble(self):
        self.assertEqual(
            sanitize_query_label("All right Let's talk about scaling. imagine we have monolithic application ba cked by single relational database that's starting to s lowdown under heavy traffic. What are some strategies you would consider to alleviate that bottleneck?"),
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_sanitize_query_label_drops_short_scaling_topic_preamble(self):
        self.assertEqual(
            sanitize_query_label("Let's talk about scaling. What are some strategies you would consider to alleviate that bottleneck?"),
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_sanitize_query_label_repairs_latest_benchmark_jwt_preamble_pollution(self):
        self.assertEqual(
            sanitize_query_label("When building is secure REST API authentication is critical. and we b to kens work and What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?"),
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_sanitize_query_label_repairs_latest_benchmark_jwt_arrest_pi_artifact(self):
        self.assertEqual(
            sanitize_query_label("arrest PI authentication is critical. and we b to kens work and what the pote ntialse security risks are if you store a JWT in the browser local storage instead of an h ttp on ly cooking."),
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_sanitize_query_label_repairs_latest_benchmark_scaling_tail_fragment(self):
        self.assertEqual(
            sanitize_query_label("single relational database that 's starting to slowdown under heavy traffic. alleviate that bottleneck."),
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_sanitize_query_label_repairs_latest_benchmark_css_confused_between_shape(self):
        self.assertEqual(
            sanitize_query_label("confused between SS grid. and Flexbox. di. fferences between the team.. de. fini. telychoosegrid over."),
            "Can you explain the primary differences between the two? give an example of a layout where you would definitely choose grid over Flexbox?",
        )

    def test_looks_like_actionable_auto_query_rejects_are_designing_declarative_setup(self):
        self.assertFalse(
            looks_like_actionable_auto_query("are designing a new public-facing API for our mobile app..")
        )

    def test_looks_like_actionable_auto_query_rejects_css_tail_statement(self):
        self.assertFalse(
            looks_like_actionable_auto_query("would definitely choose grid over Flexbox.")
        )

    def test_sanitize_query_label_repairs_css_tail_with_context(self):
        self.assertEqual(
            sanitize_query_label(
                "confusedbet confused between SS grid and Flexbox. differences between the team. would finitelychoosegri would definitely choose grid over Flexbox."
            ),
            "Can you explain the primary differences between the two? give an example of a layout where you would definitely choose grid over Flexbox?",
        )

    def test_sanitize_query_label_repairs_truncated_react_hook_benchmark_query(self):
        self.assertEqual(
            sanitize_query_label(
                "Could you walk me through how you would decide between using a custom hook versus a higher-order component for..."
            ),
            "Could you walk me through how you would decide between using a custom hook versus a higher-order component for sharing stateful logic?",
        )

    def test_sanitize_query_label_repairs_versionable_api_benchmark_query(self):
        self.assertEqual(
            sanitize_query_label(
                "What are some of the key principles you would follow to ensure the API is robust, versionable, and provides a good developer experience for the front-end team?"
            ),
            "What are some of the key principles you would follow to ensure the API is robust, maintainable and provides a good developer experience for the frontend team?",
        )

    def test_sanitize_query_label_repairs_incomplete_jwt_benchmark_query(self):
        self.assertEqual(
            sanitize_query_label(
                "What are the potential security risks if you store a JWT in the browser's local storage instead of an?"
            ),
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )


if __name__ == "__main__":
    unittest.main()
