from unittest import TestCase

from learningscenariogenerators import OntologyGenerator
from learningscenariogenerators.exception import \
    OntologyConstraintsViolatedException
from learningscenariogenerators.simulatedannealing import \
    SimulatedAnnealingLearningScenario1Generator


class TestLearningScenario1Generator(TestCase):
    def test_init_num_examples(self):
        num_pos_examples = 20
        num_neg_examples = 20
        _num_all_examples = num_pos_examples + num_neg_examples

        # i.e. how deeply the existential restriction is nested
        nesting_depth = 5

        min_num_overall_examples = \
            (nesting_depth * _num_all_examples) + _num_all_examples

        try:
            SimulatedAnnealingLearningScenario1Generator(
                num_pos_examples=num_pos_examples,
                num_neg_examples=num_neg_examples,
                num_overall_individuals=min_num_overall_examples + 10,
                existential_nesting_depth=nesting_depth)

            self.assertTrue(True)

        except OntologyConstraintsViolatedException:
            self.fail(
                "There should be no exception being thrown here (first "
                "occurrence)")

        try:
            SimulatedAnnealingLearningScenario1Generator(
                num_pos_examples=num_pos_examples,
                num_neg_examples=num_neg_examples,
                num_overall_individuals=min_num_overall_examples,
                existential_nesting_depth=nesting_depth)

            self.assertTrue(True)

        except OntologyConstraintsViolatedException:
            self.fail(
                "There should be no exception being thrown here (second "
                "occurrence)")

        with self.assertRaises(OntologyConstraintsViolatedException):
            SimulatedAnnealingLearningScenario1Generator(
                num_pos_examples=num_pos_examples,
                num_neg_examples=num_neg_examples,
                num_overall_individuals=min_num_overall_examples - 1,
                existential_nesting_depth=nesting_depth)

    def test_init_num_obj_properties(self):
        """
        As we do not want to let object properties appear multiple times in an
        existential restriction like, e.g.,

          prop_1 some (prop_2 some ( ... prop_n some Cls23 ...))

        i.e. all prop_i (i = 1, 2, ..., n) should be different from each other
        (which eases certain implementation concerns), the number of properties
        should at least be as big as the nesting depth of the existential
        restriction.
        """

        try:
            SimulatedAnnealingLearningScenario1Generator(
                num_object_properties=20,
                existential_nesting_depth=2)

            self.assertTrue(True)

        except OntologyConstraintsViolatedException:
            self.fail(
                "There should be no exception being thrown here (first "
                "occurrence)")

        try:
            SimulatedAnnealingLearningScenario1Generator(
                num_object_properties=2,
                existential_nesting_depth=2)

            self.assertTrue(True)

        except OntologyConstraintsViolatedException:
            self.fail(
                "There should be no exception being thrown here (second "
                "occurrence)")

        with self.assertRaises(OntologyConstraintsViolatedException):
            SimulatedAnnealingLearningScenario1Generator(
                num_object_properties=1,
                existential_nesting_depth=2)

    def test_generate_target_obj_prop_nesting(self):
        num_obj_properties = 20

        # i.e. how deeply the existential restriction is nested
        nesting_depth = 5

        scenario_generator = SimulatedAnnealingLearningScenario1Generator(
            num_pos_examples=10,  # doesn't matter
            num_neg_examples=10,  # doesn't matter
            num_overall_individuals=500,  # doesn't matter
            num_object_properties=num_obj_properties,
            existential_nesting_depth=nesting_depth)

        ont_generator = OntologyGenerator()
        for i in range(num_obj_properties):
            ont_generator.add_new_object_property()

        obj_properties = ont_generator.object_properties

        target_existential_obj_prop_nesting = \
            scenario_generator._generate_target_obj_prop_nesting(ont_generator)

        self.assertEqual(
            len(set(target_existential_obj_prop_nesting)), nesting_depth)

        for obj_prop in target_existential_obj_prop_nesting:
            self.assertTrue(obj_prop in obj_properties)