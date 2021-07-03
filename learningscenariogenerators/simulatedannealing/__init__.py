from random import randint, gauss, choice, shuffle, random
from typing import List, Set

from lorem import word, get_word
from morelianoctua.model import OWLOntology, OWLAxiom
from morelianoctua.model.axioms.assertionaxiom import OWLClassAssertionAxiom, \
    OWLObjectPropertyAssertionAxiom, OWLDataPropertyAssertionAxiom
from morelianoctua.model.axioms.classaxiom import OWLSubClassOfAxiom
from morelianoctua.model.axioms.declarationaxiom import \
    OWLClassDeclarationAxiom, OWLObjectPropertyDeclarationAxiom
from morelianoctua.model.axioms.owldatapropertyaxiom import \
    OWLDataPropertyDomainAxiom, OWLDataPropertyRangeAxiom
from morelianoctua.model.axioms.owlobjectpropertyaxiom import \
    OWLObjectPropertyDomainAxiom, OWLObjectPropertyRangeAxiom
from morelianoctua.model.objects.classexpression import OWLClass, \
    OWLObjectSomeValuesFrom, OWLObjectIntersectionOf, OWLClassExpression
from morelianoctua.model.objects.datarange import OWLDatatype
from morelianoctua.model.objects.individual import OWLNamedIndividual
from morelianoctua.model.objects.property import OWLObjectProperty, \
    OWLDataProperty
from morelianoctua.reasoning.owllinkreasoner import OWLLinkReasoner
from rdflib import URIRef, XSD, Literal, OWL

from learningscenariogenerators import LearningScenarioGenerator, \
    LearningScenario, OntologyGenerator
from learningscenariogenerators.exception import \
    OntologyConstraintsViolatedException


# class SimulatedAnnealingLearningScenario1Generator(LearningScenarioGenerator):
#     """
#     Generates a class expression learning scenario showing the potentials of
#     using a simulated annealing approach.
#
#     For class expression learning in description logics we focus on the top-down
#     refinement approach which structures the search space based on iteratively
#     refining current best candidate class expressions to more specific ones,
#     e.g. by considering the class hierarchy of an ontology, or combining
#     existing (complex) candidate class expressions or atomic classes
#     conjunctively etc. (We may also allow upward refinements by introducing
#     disjunctive combinations, though.) This allows a systematic exploration of
#     the search space. However, since this allows a wide range of possible
#     refinements of a single candidate class expression in a single refinement
#     step the search space is huge.
#
#     When balancing
#
#     - exploitation (i.e. the greedy search following a -- mostly
#       deterministic -- heuristic), and
#     - exploration (i.e. providing a good coverage of the whole search space
#       without strict guidance, e.g. by randomly generating class expressions)
#
#     the simulated annealing approach starts in a highly explorative setting and
#     'cools down' to a more exploitative setting after some time or whenever the
#     score (e.g. accuracy, f1-score, ...) of a learned candidate class expression
#     increases. In an adaptive simulated annealing setting the whole procedure
#     may 'heat up' in case the learning process is stuck on a plateau or in a
#     local optimum, i.e. whenever there was no increase in the score (no better
#     candidate class expression could be found) for a certain amount of time.
#
#     So a learning scenario which would highly benefit from a simulated
#     annealing-based class expression learning approach would have the following
#     characteristics
#
#     C1) The target concept has multiple parts (e.g. disjunctively connected)
#         which is not findable via straight greedy downward refinement
#     C2) The learning scenario induces a plateau during the concept learning
#         process which cannot be passed by a myopic greedy learning approach.
#         This happens e.g. when all the positive examples are instances of a
#         class expression
#
#             hasComponent some GoodComponent
#
#         and all the negative examples are instances of the class expression
#
#             hasComponent some BadComponent
#
#         During the systematic downward refinement process at some point we may
#         create the candidate class expression
#
#            hasComponent some Thing
#
#         which is only one refinement step away from finding the perfect
#         explanation for the positive examples (i.e. being instance of
#         hasComponent some GoodComponent -- assuming GoodComponent and
#         BadComponent are disjoint classes). However hasComponent some Thing
#         would possibly be considered too weak since it only gives an accuracy
#         score of 0.5 (assuming the number of positive and negative examples is
#         balanced).
#     """
#     def __init__(
#             self,
#             num_pos_examples: int = 50,
#             num_neg_examples: int = 50,
#             num_classes: int = 30,
#             num_object_properties: int = 10,
#             num_data_properties: int = 5,
#             num_overall_individuals: int = 500,
#             existential_nesting_depth: int = 5):
#
#         self.num_pos_examples = num_pos_examples
#         self.num_neg_examples = num_neg_examples
#         self.num_classes = num_classes
#         self.num_object_properties = num_object_properties
#         self.num_data_properties = num_data_properties
#         self.num_overall_individuals = num_overall_individuals
#         self.existential_nesting_depth = existential_nesting_depth
#
#         self.num_non_example_individuals = \
#             num_overall_individuals - num_pos_examples - num_neg_examples
#         assert self.num_non_example_individuals >= num_pos_examples + \
#             num_neg_examples
#
#         self.ontology_prefix = \
#             'http://dl-learner.org/ontology%06i#' % randint(1, 999999)
#
#         self._cls_cntr = 0
#         self._obj_prop_cntr = 0
#         self._data_prop_cntr = 0
#         self._indiv_cntr = 0
#         self._literal_cntr = 0
#         self._possible_data_property_ranges = [
#             OWLDatatype(XSD.int),
#             OWLDatatype(XSD.double),
#             OWLDatatype(XSD.string)]
#
#     @staticmethod
#     def _next_literal(dtype):
#         if dtype.iri == XSD.string:
#             return Literal(get_word(), None, XSD.string)
#         elif dtype.iri == XSD.int:
#             return Literal(randint(1, 23), None, XSD.int)
#         elif dtype.iri == XSD.double:
#             return Literal(gauss(23, 5), None, XSD.double)
#         else:
#             raise Exception(f'Unhandled data type {str(dtype)}')
#
#     def _next_cls_iri(self):
#         self._cls_cntr += 1
#         return URIRef(self.ontology_prefix + f'Cls{self._cls_cntr}')
#
#     def _next_obj_prop_iri(self):
#         self._obj_prop_cntr += 1
#         return URIRef(self.ontology_prefix + f'objProp{self._obj_prop_cntr}')
#
#     def _next_data_prop_iri(self):
#         self._data_prop_cntr += 1
#         return URIRef(self.ontology_prefix + f'dataProp{self._data_prop_cntr}')
#
#     def _generate_atomic_classes(self) -> List[OWLClass]:
#         classes = []
#
#         for i in range(self.num_classes):
#             cls = OWLClass(self._next_cls_iri())
#             classes.append(cls)
#
#         return classes
#
#     def _generate_obj_props(self) -> List[OWLObjectProperty]:
#         obj_props = []
#
#         for i in range(self.num_object_properties):
#             obj_prop = OWLObjectProperty(self._next_obj_prop_iri())
#             obj_props.append(obj_prop)
#
#         return obj_props
#
#     def _generate_data_props(self) -> List[OWLDataProperty]:
#         data_props = []
#
#         for i in range(self.num_data_properties):
#             data_prop = OWLDataProperty(self._next_data_prop_iri())
#             data_props.append(data_prop)
#
#         return data_props
#
#     def _generate_individuals(
#             self,
#             number_of_indivs_to_generate: int,
#             local_part_prefix: str) -> List[OWLNamedIndividual]:
#
#         individuals = []
#
#         for i in range(number_of_indivs_to_generate):
#             self._indiv_cntr += 1
#             indiv = OWLNamedIndividual(
#                 self.ontology_prefix +
#                 local_part_prefix +
#                 str(self._indiv_cntr))
#
#             individuals.append(indiv)
#
#         return individuals
#
#     def _get_all_sub_classes(self, cls_hierarchy: dict, super_cls: OWLClass) \
#             -> List[OWLClass]:
#         sub_classes = cls_hierarchy[super_cls].copy()
#
#         for sub_cls in cls_hierarchy[super_cls]:
#             sub_classes += self._get_all_sub_classes(cls_hierarchy, sub_cls)
#
#         return sub_classes
#
#     def generate_scenario(self) -> LearningScenario:
#         all_classes: List[OWLClass] = self._generate_atomic_classes()
#
#         all_obj_properties: List[OWLObjectProperty] = self._generate_obj_props()
#         all_datatype_properties: List[OWLDataProperty] = \
#             self._generate_data_props()
#
#         axioms = set()
#         obj_property_domains = {}
#         obj_property_ranges = {}
#         data_property_domains = {}
#         data_property_ranges = {}
#         has_instances = {}
#
#         target_atomic_cls = choice(all_classes)
#         target_obj_prop_nesting_seq: List[OWLObjectProperty] = []
#
#         for i in range(self.existential_nesting_depth):
#             target_obj_prop_nesting_seq.append(choice(all_obj_properties))
#         last_target_obj_prop_in_seq_pos_range = None
#         last_target_obj_prop_in_seq_neg_range = None
#
#         owl_thing = OWLClass(OWL.Thing)
#         super_classes = [owl_thing]
#         class_hierarchy = {owl_thing: []}
#
#         for cls in all_classes:
#             axioms.add(OWLClassDeclarationAxiom(cls))
#             super_class = choice(super_classes)
#             axioms.add(OWLSubClassOfAxiom(cls, super_class))
#
#             if class_hierarchy.get(cls) is None:
#                 class_hierarchy[cls] = []
#
#             class_hierarchy[super_class].append(cls)
#             super_classes.append(cls)
#
#         for object_property in all_obj_properties:
#             axioms.add(OWLObjectPropertyDeclarationAxiom(object_property))
#
#             if object_property == target_obj_prop_nesting_seq[-1]:
#                 possible_domains = self._get_all_sub_classes(
#                     class_hierarchy, target_atomic_cls)
#                 possible_domains.append(target_atomic_cls)
#         #
#         #         domain = choice(possible_domains)
#         #
#         #         obj_property_domains[object_property] = domain
#         #         axioms.add(
#         #             OWLObjectPropertyDomainAxiom(object_property, domain))
#         #
#         #         # a range class is to be found that has at least one sibling
#         #         # class which then would become the target class of the negative
#         #         # examples
#         #         range_candidates = all_classes.copy()
#         #         shuffle(range_candidates)
#         #         range_cls = None
#         #         for range_candidate in range_candidates:
#         #             if len(class_hierarchy[range_candidate]) > 1:
#         #                 range_cls = range_candidate
#         #                 break
#         #
#         #         if range_cls is None:
#         #             raise Exception(
#         #                 'Generated class hierarchy doesn\'t allow the creation '
#         #                 'of a proper learning problem. Please re-run.')
#         #
#         #         obj_property_ranges[object_property] = range_cls
#         #         axioms.add(
#         #             OWLObjectPropertyRangeAxiom(object_property, range_cls))
#         #
#         #         pos_range_super_cls = class_hierarchy[range_cls][0]
#         #         neg_range_super_cls = class_hierarchy[range_cls][1]
#         #
#         #         pos_range_cls_candidates = \
#         #             self._get_all_sub_classes(
#         #                 class_hierarchy, pos_range_super_cls)
#         #         pos_range_cls_candidates.append(pos_range_super_cls)
#         #
#         #         target_obj_prop_pos_range = choice(pos_range_cls_candidates)
#         #
#         #         neg_range_cls_candidates = \
#         #             self._get_all_sub_classes(
#         #                 class_hierarchy, neg_range_super_cls)
#         #         neg_range_cls_candidates.append(neg_range_super_cls)
#         #
#         #         target_obj_prop_neg_range = choice(neg_range_cls_candidates)
#         #
#         #     else:
#         #         domain = choice(all_classes)
#         #         obj_property_domains[object_property] = domain
#         #         axioms.add(
#         #             OWLObjectPropertyDomainAxiom(object_property, domain))
#         #
#         #         range_cls = choice(all_classes)
#         #         obj_property_ranges[object_property] = range_cls
#         #         axioms.add(
#         #             OWLObjectPropertyRangeAxiom(object_property, range_cls))
#         #
#         # for data_prop in all_datatype_properties:
#         #     domain = choice(all_classes)
#         #     data_property_domains[data_prop] = domain
#         #     axioms.add(OWLDataPropertyDomainAxiom(data_prop, domain))
#         #
#         #     data_range = choice(self._possible_data_property_ranges)
#         #     data_property_ranges[data_prop] = data_range
#         #     axioms.add(OWLDataPropertyRangeAxiom(data_prop, data_range))
#         #
#         pos_examples = \
#             self._generate_individuals(self.num_pos_examples, 'pos_indiv')
#         neg_examples = \
#             self._generate_individuals(self.num_neg_examples, 'neg_indiv')
#         other_individuals = self._generate_individuals(
#             self.num_non_example_individuals, 'indiv')
#         shuffle(other_individuals)
#         #
#         # # will be used to pick 'value individuals' for object property
#         # # assertions
#         # all_individuals = pos_examples + neg_examples + other_individuals
#         # shuffle(all_individuals)
#         #
#         # pos_obj_indiv_cls_candidates = self._get_all_sub_classes(
#         #     class_hierarchy, target_obj_prop_pos_range)
#         # pos_obj_indiv_cls_candidates.append(target_obj_prop_pos_range)
#         #
#         # for pos_example in pos_examples:
#         #     axioms.add(OWLClassAssertionAxiom(pos_example, target_atomic_cls))
#         #
#         #     if has_instances.get(target_atomic_cls) is None:
#         #         has_instances[target_atomic_cls] = []
#         #     has_instances[target_atomic_cls].append(pos_example)
#         #
#         #     object_individual = other_individuals.pop(0)
#         #
#         #     axioms.add(
#         #         OWLObjectPropertyAssertionAxiom(
#         #             pos_example, target_obj_prop, object_individual))
#         #
#         #     obj_indiv_cls = choice(pos_obj_indiv_cls_candidates)
#         #     axioms.add(OWLClassAssertionAxiom(object_individual, obj_indiv_cls))
#         #
#         #     data_property = choice(all_datatype_properties)
#         #     data_prop_range = data_property_ranges[data_property]
#         #     literal = self._next_literal(data_prop_range)
#         #     axioms.add(
#         #         OWLDataPropertyAssertionAxiom(
#         #             pos_example, data_property, literal))
#         #
#         # neg_obj_indiv_cls_candidates = self._get_all_sub_classes(
#         #     class_hierarchy, target_obj_prop_neg_range)
#         # neg_obj_indiv_cls_candidates.append(target_obj_prop_neg_range)
#         #
#         # for neg_example in neg_examples:
#         #     axioms.add(OWLClassAssertionAxiom(neg_example, target_atomic_cls))
#         #
#         #     if has_instances.get(target_atomic_cls) is None:
#         #         has_instances[target_atomic_cls] = []
#         #     has_instances[target_atomic_cls].append(neg_example)
#         #
#         #     object_individual = other_individuals.pop(0)
#         #
#         #     axioms.add(
#         #         OWLObjectPropertyAssertionAxiom(
#         #             neg_example, target_obj_prop, object_individual))
#         #
#         #     obj_indiv_cls = choice(neg_obj_indiv_cls_candidates)
#         #     axioms.add(OWLClassAssertionAxiom(object_individual, obj_indiv_cls))
#         #
#         #     data_property = choice(all_datatype_properties)
#         #     data_prop_range = data_property_ranges[data_property]
#         #     literal = self._next_literal(data_prop_range)
#         #     axioms.add(
#         #         OWLDataPropertyAssertionAxiom(
#         #             neg_example, data_property, literal))
#         #
#         # for indiv in other_individuals:
#         #     indv_cls = choice(all_classes)
#         #     axioms.add(OWLClassAssertionAxiom(indiv, indv_cls))
#         #
#         #     # indiv appears as subject individual with random obj property
#         #     if random() > 0.5:
#         #         object_property = choice(all_obj_properties)
#         #         obj_prop_range_cls = obj_property_ranges[object_property]
#         #
#         #         if has_instances.get(obj_prop_range_cls) is None:
#         #             object_indiv = choice(all_individuals)
#         #             axioms.add(
#         #                 OWLClassAssertionAxiom(
#         #                     object_indiv, obj_prop_range_cls))
#         #
#         #         else:
#         #             object_indiv = choice(has_instances[obj_prop_range_cls])
#         #
#         #         axioms.add(
#         #             OWLObjectPropertyAssertionAxiom(
#         #                 indiv, object_property, object_indiv))
#         #
#         #     # indiv appears as object individual with random obj property
#         #     if random() > 0.5:
#         #         object_property = choice(all_obj_properties)
#         #         domain_cls = obj_property_domains[object_property]
#         #
#         #         if has_instances.get(domain_cls) is None:
#         #             subject_indiv = choice(all_individuals)
#         #             axioms.add(
#         #                 OWLClassAssertionAxiom(subject_indiv, domain_cls))
#         #         else:
#         #             subject_indiv = choice(has_instances[domain_cls])
#         #
#         #         axioms.add(
#         #             OWLObjectPropertyAssertionAxiom(
#         #                 subject_indiv, object_property, indiv))
#         #
#         #     # indiv appears as subject individual with random data property
#         #     if random() > 0.5:
#         #         data_property = choice(all_datatype_properties)
#         #         data_prop_range = data_property_ranges[data_property]
#         #
#         #         literal = self._next_literal(data_prop_range)
#         #
#         #         axioms.add(
#         #             OWLDataPropertyAssertionAxiom(
#         #                 indiv, data_property, literal))
#
#         ont = OWLOntology(prefix_declarations={}, axioms=axioms)
#
#         # g = ont.as_rdf_graph()
#         # g.serialize(open('/tmp/dbg.ttl', 'wb'), format='turtle')
#
#         return LearningScenario(
#             {p for p in pos_examples},
#             {n for n in neg_examples},
#             OWLObjectIntersectionOf(
#                 target_atomic_cls,
#                 OWLObjectSomeValuesFrom(target_obj_prop, OWLClass(str(OWL.Thing)))),
#             ont)


class SimulatedAnnealingLearningScenario1Generator(LearningScenarioGenerator):
    """
    Generates a class expression learning scenario showing the potentials of
    using a simulated annealing approach.

    For class expression learning in description logics we focus on the top-down
    refinement approach which structures the search space based on iteratively
    refining current best candidate class expressions to more specific ones,
    e.g. by considering the class hierarchy of an ontology, or combining
    existing (complex) candidate class expressions or atomic classes
    conjunctively etc. (We may also allow upward refinements by introducing
    disjunctive combinations, though.) This allows a systematic exploration of
    the search space. However, since this allows a wide range of possible
    refinements of a single candidate class expression in a single refinement
    step the search space is huge.

    When balancing

    - exploitation (i.e. the greedy search following a -- mostly
      deterministic -- heuristic), and
    - exploration (i.e. providing a good coverage of the whole search space
      without strict guidance, e.g. by randomly generating class expressions)

    the simulated annealing approach starts in a highly explorative setting and
    'cools down' to a more exploitative setting after some time or whenever the
    score (e.g. accuracy, f1-score, ...) of a learned candidate class expression
    increases. In an adaptive simulated annealing setting the whole procedure
    may 'heat up' in case the learning process is stuck on a plateau or in a
    local optimum, i.e. whenever there was no increase in the score (no better
    candidate class expression could be found) for a certain amount of time.

    So a learning scenario which would highly benefit from a simulated
    annealing-based class expression learning approach would have the following
    characteristics

    C1) The target concept has multiple parts (e.g. disjunctively connected)
        which is not findable via straight greedy downward refinement
    C2) The learning scenario induces a plateau during the concept learning
        process which cannot be passed by a myopic greedy learning approach.
        This happens e.g. when all the positive examples are instances of a
        class expression

            hasComponent some GoodComponent

        and all the negative examples are instances of the class expression

            hasComponent some BadComponent

        During the systematic downward refinement process at some point we may
        create the candidate class expression

           hasComponent some Thing

        which is only one refinement step away from finding the perfect
        explanation for the positive examples (i.e. being instance of
        hasComponent some GoodComponent -- assuming GoodComponent and
        BadComponent are disjoint classes). However hasComponent some Thing
        would possibly be considered too weak since it only gives an accuracy
        score of 0.5 (assuming the number of positive and negative examples is
        balanced).
    """
    def __init__(
            self,
            num_pos_examples: int = 50,
            num_neg_examples: int = 50,
            num_classes: int = 30,
            num_object_properties: int = 10,
            num_data_properties: int = 5,
            num_overall_individuals: int = 500,
            existential_nesting_depth: int = 2):

        self.num_pos_examples = num_pos_examples
        self.num_neg_examples = num_neg_examples
        self.num_classes = num_classes
        self.num_object_properties = num_object_properties
        self.num_data_properties = num_data_properties
        self.num_overall_individuals = num_overall_individuals
        self.existential_nesting_depth = existential_nesting_depth

        self.num_non_example_individuals = \
            num_overall_individuals - num_pos_examples - num_neg_examples

        if self.num_non_example_individuals < existential_nesting_depth * \
                (num_pos_examples + num_neg_examples):
            raise OntologyConstraintsViolatedException(
                f'The number of overall examples must be greater than or equal '
                f'to (existential_nesting_depth(={existential_nesting_depth}) '
                f'* the number of positive and negative examples) + (the '
                f'number of positive and negative examples) as we do not want '
                f'to re-use non-example individuals in relation paths')

        if num_object_properties < existential_nesting_depth:
            raise OntologyConstraintsViolatedException(
                'The number of object properties should be greater than or '
                'equal to the nesting depth as we do not allow object '
                'properties appearing multiple times in the object property '
                'nesting sequence of the existential restriction'
            )

    def _generate_target_obj_prop_nesting(
            self, ont_generator: OntologyGenerator) -> List[OWLObjectProperty]:

        target_obj_prop_nesting_seq = []

        for i in range(self.existential_nesting_depth):
            while True:
                __obj_prop = ont_generator.pick_random_object_property()

                if __obj_prop not in target_obj_prop_nesting_seq:
                    break

            target_obj_prop_nesting_seq.append(__obj_prop)

        return target_obj_prop_nesting_seq

    def _get_final_filler(self, ont_generator: OntologyGenerator) -> OWLClass:
        """
        The final filler class of the nested existential restriction we want to
        generate here should have the following properties

        - there should be at least two subclasses, holding the positive and
          negative examples, respectively
        - the cardinality of the union of the class's parents, the class's
          siblings, and the siblings children should be at least as big as the
          nesting depth s.t. we can choose domain/range classes for object
          properties from the existential restriction sequence which are not
          inside the sub-tree of the filler class
        - That's it? -.-
        """

        # Loop until we find a class that has at least two subclasses and there
        # are more than self.existential_nesting_depth classes left in the
        # class hierarchy outside the subclass subtree of that class
        while True:
            filler_candidate = ont_generator.pick_random_class()

            if len(ont_generator.get_all_direct_sub_classes(
                    filler_candidate)) >= 2 and \
                    len(ont_generator.get_classes_from_complement_of_subtree(
                        filler_candidate)) > self.existential_nesting_depth:
                break

        return filler_candidate

    def _remove_examples_from_indivs(
            self,
            individuals: List[OWLNamedIndividual],
            examples: List[OWLNamedIndividual]):

        for example in examples:
            individuals.remove(example)

    def generate_scenario(self) -> LearningScenario:
        ontology_generator = OntologyGenerator()

        for i in range(self.num_classes):
            ontology_generator.add_new_class()

        for i in range(self.num_object_properties):
            ontology_generator.add_new_object_property()

        for i in range(self.num_data_properties):
            ontology_generator.add_new_data_property()

        ontology_generator.init_random_class_hierarchy()

        pos_examples = []
        for i in range(self.num_pos_examples):
            pos_example = ontology_generator.add_new_individual('pos_indiv')
            pos_examples.append(pos_example)

        neg_examples = []
        for i in range(self.num_neg_examples):
            neg_example = ontology_generator.add_new_individual('neg_indiv')
            neg_examples.append(neg_example)

        non_example_individuals = []
        for i in range(self.num_non_example_individuals):
            indiv = ontology_generator.add_new_individual()
            non_example_individuals.append(indiv)

        ext_restr_final_filler_class = \
            self._get_final_filler(ontology_generator)

        _all_direct_filler_subclasses = \
            ontology_generator.get_all_direct_sub_classes(
                ext_restr_final_filler_class)
        shuffle(_all_direct_filler_subclasses)

        pos_examples_ext_restr_final_filler_class = \
            _all_direct_filler_subclasses.pop(0)
        neg_examples_ext_restr_final_filler_class = \
            _all_direct_filler_subclasses.pop(0)

        target_obj_prop_nesting_seq = \
            self._generate_target_obj_prop_nesting(ontology_generator)

        target_obj_prop_nesting_seq_domains = []
        target_obj_prop_nesting_seq_ranges = []

        # Set domains and ranges for all the object properties involved in the
        # nested existential restriction. Class candidates are all those classes
        # - not equal to the filler of the final existential restriction in the
        #   existential restriction nesting
        # - not being a subclass of the filler of the final existential
        #   restriction in the existential restriction nesting
        cls_candidates = \
            ontology_generator.get_classes_from_complement_of_subtree(
                ext_restr_final_filler_class)
        shuffle(cls_candidates)

        last_range = None
        #                           all except last obj prop
        for i in range(0, len(target_obj_prop_nesting_seq)-1):
            obj_prop = target_obj_prop_nesting_seq[i]

            if last_range is None:
                domain = cls_candidates.pop(0)
            else:
                domain = last_range

            rnge = cls_candidates.pop(0)

            target_obj_prop_nesting_seq_domains.append(domain)
            target_obj_prop_nesting_seq_ranges.append(rnge)

            ontology_generator.set_object_property_domain_and_range(
                obj_prop, domain, rnge)

            last_range = rnge

        ontology_generator.set_object_property_domain_and_range(
            target_obj_prop_nesting_seq[-1],
            last_range,
            ext_restr_final_filler_class)
        target_obj_prop_nesting_seq_domains.append(last_range)
        target_obj_prop_nesting_seq_ranges.append(ext_restr_final_filler_class)

        assert len(target_obj_prop_nesting_seq) == \
            len(target_obj_prop_nesting_seq_domains)

        assert len(target_obj_prop_nesting_seq_domains) == \
            len(target_obj_prop_nesting_seq_ranges)

        # build target class expression
        target_cls_expression = pos_examples_ext_restr_final_filler_class

        for i in range(len(target_obj_prop_nesting_seq)-1, -1, -1):
            obj_prop = target_obj_prop_nesting_seq[i]
            target_cls_expression = \
                OWLObjectSomeValuesFrom(obj_prop, target_cls_expression)

        # build target negative class expression
        target_neg_cls_expression = neg_examples_ext_restr_final_filler_class

        for i in range(len(target_obj_prop_nesting_seq) - 1, -1, -1):
            obj_prop = target_obj_prop_nesting_seq[i]
            target_neg_cls_expression = \
                OWLObjectSomeValuesFrom(obj_prop, target_neg_cls_expression)

        # set domain and ranges for remaining object properties
        for obj_prop in ontology_generator.object_properties:
            if obj_prop in target_obj_prop_nesting_seq:
                continue  # We already set domain/range for those

            # FIXME: Might be this is too naive and we have to really only pick
            # classes not being subclasses of any of the domains/ranges assigned
            # to the obj properties from the existential restriction
            domain = ontology_generator.pick_random_class()
            rnge = ontology_generator.pick_random_class()

            ontology_generator.set_object_property_domain_and_range(
                obj_prop, domain, rnge)

        pos_ext_restr_final_filler_classes = \
            ontology_generator.get_all_sub_classes(
                pos_examples_ext_restr_final_filler_class)

        neg_examples_ext_restr_final_filler_classes = \
            ontology_generator.get_all_sub_classes(
                neg_examples_ext_restr_final_filler_class)

        # for data_prop in ontology_generator.data_properties:
        #     domain_cls = ontology_generator.pick_random_class()
        #     range_dtype = ontology_generator.pick_random_datatype()
        #
        #     ontology_generator.set_data_property_domain_and_range(
        #         data_prop, domain_cls, range_dtype)

        tmp_individuals = ontology_generator.individuals.copy()
        _len_before = len(tmp_individuals)
        self._remove_examples_from_indivs(tmp_individuals, pos_examples)
        self._remove_examples_from_indivs(tmp_individuals, neg_examples)
        _len_after = len(tmp_individuals)
        assert _len_before == _len_after + len(pos_examples) + len(neg_examples)

        shuffle(tmp_individuals)

        prop_add_probability = 0.5

        for pos_example in pos_examples:
            print(f'pos example: {pos_example}')
            tmp_dom_indiv = pos_example
            domain_super_cls = target_obj_prop_nesting_seq_domains[0]
            domain = choice(
                ontology_generator.get_all_sub_classes(domain_super_cls))
            ontology_generator.add_instance(tmp_dom_indiv, domain)

            for i in range(len(target_obj_prop_nesting_seq)-1):
                obj_prop = target_obj_prop_nesting_seq[i]

                range_super_cls = target_obj_prop_nesting_seq_ranges[i]
                rnge = choice(
                    ontology_generator.get_all_sub_classes(range_super_cls))

                tmp_range_indiv = tmp_individuals.pop(0)
                ontology_generator.add_instance(tmp_range_indiv, rnge)

                ontology_generator.add_axiom(
                    OWLObjectPropertyAssertionAxiom(
                        tmp_dom_indiv, obj_prop, tmp_range_indiv))
                tmp_dom_indiv = tmp_range_indiv

            # last filler individual should not be instance of the range cls
            # but of the more special positive filler class
            obj_prop = target_obj_prop_nesting_seq[-1]
            range_super_cls = pos_examples_ext_restr_final_filler_class
            rnge = choice(
                ontology_generator.get_all_sub_classes(range_super_cls))

            tmp_range_indiv = tmp_individuals.pop(0)
            ontology_generator.add_instance(tmp_range_indiv, rnge)

            ontology_generator.add_axiom(
                OWLObjectPropertyAssertionAxiom(
                    tmp_dom_indiv, obj_prop, tmp_range_indiv))

            properties_i_gave_up_with = []
            # add further random assertions (pos_example, propX, other_indiv)
            while random() > prop_add_probability:
                possible_obj_props = \
                    ontology_generator.get_object_properties_by_domain(domain)

                # should always work as at least target_obj_prop_nesting_seq[0]
                # has domain as domain
                obj_prop = choice(possible_obj_props)
                if obj_prop in properties_i_gave_up_with:
                    continue

                print(f'prop: {obj_prop}')

                max_number_of_attempts = 10

                attempt_cntr = 0
                while True:
                    if attempt_cntr == max_number_of_attempts:
                        break

                    rnge = ontology_generator.get_random_range_class(obj_prop)
                    print(f'range: {rnge}')

                    obj_indiv = ontology_generator.\
                        pick_random_individual_by_cls(rnge)

                    attempt_cntr += 1

                    if obj_indiv is None:
                        print('No instance found')
                        continue
                    break

                if attempt_cntr == max_number_of_attempts:
                    print(f'Giving up after {attempt_cntr} attempts')
                    properties_i_gave_up_with.append(obj_prop)
                    continue

                ontology_generator.add_axiom(
                    OWLObjectPropertyAssertionAxiom(
                        pos_example, obj_prop, obj_indiv))

            properties_i_gave_up_with = []
            # add further random assertions (other_indiv, propX, pos_example)
            while random() > prop_add_probability:
                possible_obj_props = \
                    ontology_generator.get_object_properties_by_range(domain)

                if len(possible_obj_props) == 0:
                    break

                obj_prop = choice(possible_obj_props)
                if obj_prop in properties_i_gave_up_with:
                    continue

                print(f'prop: {obj_prop}')

                max_number_of_attempts = 10

                attempt_cntr = 0
                while True:
                    if attempt_cntr == max_number_of_attempts:
                        break

                    print(f'domain: {domain}')
                    domain = ontology_generator.get_random_domain_class(
                        obj_prop)

                    subj_indiv = ontology_generator.\
                        pick_random_individual_by_cls(domain)

                    attempt_cntr += 1
                    if subj_indiv is None:
                        print('No instance found')
                        continue
                    break

                if attempt_cntr == max_number_of_attempts:
                    print(f'Giving up after {attempt_cntr} attempts')
                    properties_i_gave_up_with.append(obj_prop)
                    continue

                ontology_generator.add_axiom(
                    OWLObjectPropertyAssertionAxiom(
                        subj_indiv, obj_prop, pos_example))
        print('Done with pos examples')

        for neg_example in neg_examples:
            print(f'neg: {neg_example}')
            tmp_dom_indiv = neg_example
            domain_super_cls = target_obj_prop_nesting_seq_domains[0]
            domain = choice(
                ontology_generator.get_all_sub_classes(domain_super_cls))
            ontology_generator.add_instance(tmp_dom_indiv, domain)

            for i in range(len(target_obj_prop_nesting_seq)-1):
                obj_prop = target_obj_prop_nesting_seq[i]

                range_super_cls = target_obj_prop_nesting_seq_ranges[i]
                rnge = choice(
                    ontology_generator.get_all_sub_classes(range_super_cls))

                tmp_range_indiv = tmp_individuals.pop(0)
                ontology_generator.add_instance(tmp_range_indiv, rnge)

                ontology_generator.add_axiom(
                    OWLObjectPropertyAssertionAxiom(
                        tmp_dom_indiv, obj_prop, tmp_range_indiv))
                tmp_dom_indiv = tmp_range_indiv

            # last filler individual should not be instance of the range cls
            # but of the more special positive filler class
            obj_prop = target_obj_prop_nesting_seq[-1]
            range_super_cls = neg_examples_ext_restr_final_filler_class
            rnge = choice(
                ontology_generator.get_all_sub_classes(range_super_cls))

            tmp_range_indiv = tmp_individuals.pop(0)
            ontology_generator.add_instance(tmp_range_indiv, rnge)

            ontology_generator.add_axiom(
                OWLObjectPropertyAssertionAxiom(
                    tmp_dom_indiv, obj_prop, tmp_range_indiv))

            properties_i_gave_up_with = []
            # add further random assertions (neg_example, propX, other_indiv)
            while random() > prop_add_probability:
                possible_obj_props = \
                    ontology_generator.get_object_properties_by_domain(domain)

                # should always work as at least target_obj_prop_nesting_seq[0]
                # has domain as domain
                obj_prop = choice(possible_obj_props)
                if obj_prop in properties_i_gave_up_with:
                    continue

                print(f'prop: {obj_prop}')

                max_number_of_attempts = 10

                attempt_cntr = 0
                while True:
                    if attempt_cntr == max_number_of_attempts:
                        break

                    rnge = ontology_generator.get_random_range_class(obj_prop)
                    print(f'range: {rnge}')

                    # if len(tmp_individuals) > 0:
                    #     obj_indiv = tmp_individuals.pop(0)
                    #     ontology_generator.add_instance(obj_indiv, rnge)
                    #
                    # else:
                    obj_indiv = ontology_generator.\
                        pick_random_individual_by_cls(rnge)

                    attempt_cntr += 1
                    if obj_indiv is None:
                        print('No instance found')
                        continue
                    break

                if attempt_cntr == max_number_of_attempts:
                    print(f'Giving up after {attempt_cntr} attempts')
                    properties_i_gave_up_with.append(obj_prop)
                    continue

                ontology_generator.add_axiom(
                    OWLObjectPropertyAssertionAxiom(
                        neg_example, obj_prop, obj_indiv))

            properties_i_gave_up_with = []
            # add further random assertions (other_indiv, propX, neg_example)
            while random() > prop_add_probability:
                possible_obj_props = \
                    ontology_generator.get_object_properties_by_range(domain)

                if len(possible_obj_props) == 0:
                    break

                obj_prop = choice(possible_obj_props)
                if obj_prop in properties_i_gave_up_with:
                    continue

                print(f'prop: {obj_prop}')

                max_number_of_attempts = 10

                attempt_cntr = 0
                while True:
                    if attempt_cntr == max_number_of_attempts:
                        break

                    domain = ontology_generator.get_random_domain_class(
                        obj_prop)
                    print(f'domain: {domain}')

                    # if len(tmp_individuals) > 0:
                    #     subj_indiv = tmp_individuals.pop(0)
                    #     ontology_generator.add_instance(subj_indiv, domain)
                    #
                    # else:
                    subj_indiv = ontology_generator.\
                        pick_random_individual_by_cls(domain)

                    attempt_cntr += 1
                    if subj_indiv is None:
                        print('No instance found')
                        continue
                    break

                if attempt_cntr == max_number_of_attempts:
                    print(f'Giving up after {attempt_cntr} attempts')
                    properties_i_gave_up_with.append(obj_prop)
                    continue

                ontology_generator.add_axiom(
                    OWLObjectPropertyAssertionAxiom(
                        subj_indiv, obj_prop, neg_example))
        print('Done with neg examples')

        for indiv in tmp_individuals:

            while True:
                obj_prop = ontology_generator.pick_random_object_property()

                if obj_prop not in target_obj_prop_nesting_seq:
                    break

            cls = ontology_generator.get_random_domain_class(obj_prop)

            ontology_generator.add_instance(indiv, cls)

            # additional assertions (indiv, someObjProp, otherIndiv)
            while random() > prop_add_probability:
                rnge = ontology_generator.get_range_cls(obj_prop)

                obj_indiv = ontology_generator.\
                    pick_random_individual_by_cls(rnge)
                if obj_indiv is None:
                    continue

                ontology_generator.add_axiom(
                    OWLObjectPropertyAssertionAxiom(indiv, obj_prop, obj_indiv))

            # additional assertions (otherIndiv, someObjProp, indiv)
            while random() > prop_add_probability:
                possible_obj_props = \
                    ontology_generator.get_object_properties_by_range(cls)

                if len(possible_obj_props) == 0:
                    break

                obj_prop = choice(possible_obj_props)

                domain = ontology_generator.get_domain_cls(obj_prop)

                subj_indiv = ontology_generator.\
                    pick_random_individual_by_cls(domain)
                if subj_indiv is None:
                    continue

                ontology_generator.add_axiom(
                    OWLObjectPropertyAssertionAxiom(
                        subj_indiv, obj_prop, indiv))
        print('Done with remaining individuals')

        # pos_bridge_individuals = []
        # for pos_example in pos_examples:
        #     while True:
        #         bridge_individual = ontology_generator.pick_random_individual()
        #
        #         if not bridge_individual == pos_example and \
        #                 not ontology_generator.is_instance_of(
        #                     bridge_individual, target_obj_prop_2_range) and \
        #                 not ontology_generator.is_instance_of(
        #                     bridge_individual, ext_restr_final_filler_class) and \
        #                 bridge_individual not in pos_examples and \
        #                 bridge_individual not in neg_examples:
        #             break
        #
        #     pos_bridge_individuals.append(bridge_individual)
        #     bridge_indiv_cls = choice(
        #         ontology_generator.get_all_sub_classes(target_obj_prop_1_range))
        #     ontology_generator.add_instance(
        #         bridge_individual, bridge_indiv_cls)
        #
        #     ontology_generator.add_axiom(
        #         OWLObjectPropertyAssertionAxiom(
        #             pos_example, target_obj_prop_1, bridge_individual))
        #
        #     while True:
        #         filler_instance = ontology_generator.pick_random_individual()
        #
        #         if not filler_instance == pos_example and \
        #                 not filler_instance == bridge_individual and \
        #                 not ontology_generator.is_instance_of(
        #                     filler_instance, target_obj_prop_1_range) and \
        #                 not ontology_generator.is_instance_of(
        #                     filler_instance, ext_restr_final_filler_class):
        #             break
        #
        #     filler_instance_cls = choice(
        #         ontology_generator.get_all_sub_classes(
        #             target_obj_prop_2_pos_range))
        #     ontology_generator.add_instance(
        #         filler_instance, filler_instance_cls)
        #
        #     ontology_generator.add_axiom(
        #         OWLObjectPropertyAssertionAxiom(
        #             bridge_individual, target_obj_prop_2, filler_instance))
        #
        #     while random() > 0.5:
        #         obj_prop = ontology_generator.pick_random_object_property()
        #
        #         # if obj_prop == target_obj_prop_1 or \
        #         #         obj_prop == target_obj_prop_2:
        #         #     continue
        #
        #         while True:
        #             object_individual = \
        #                 ontology_generator.pick_random_individual()
        #
        #             if object_individual != pos_example:
        #                 break
        #
        #         while True:
        #             obj_indiv_cls = \
        #                 ontology_generator.get_random_range_class(obj_prop)
        #
        #             is_pos_target_2_range_sub_class = \
        #                 obj_indiv_cls in ontology_generator.get_all_sub_classes(
        #                     target_obj_prop_2_pos_range)
        #             is_neg_target_2_range_sub_class = \
        #                 obj_indiv_cls in ontology_generator.get_all_sub_classes(
        #                     target_obj_prop_2_neg_range)
        #
        #             if not is_pos_target_2_range_sub_class and \
        #                     not is_neg_target_2_range_sub_class:
        #                 break
        #
        #         ontology_generator.add_instance(
        #             object_individual, obj_indiv_cls)
        #         ontology_generator.add_axiom(
        #             OWLObjectPropertyAssertionAxiom(
        #                 pos_example, obj_prop, object_individual))
        #
        #     while random() > 0.5:
        #         obj_prop = ontology_generator.pick_random_object_property()
        #
        #         if obj_prop == target_obj_prop_1 or \
        #                 obj_prop == target_obj_prop_2:
        #             continue
        #
        #         while True:
        #             subject_individual = \
        #                 ontology_generator.pick_random_individual()
        #
        #             if subject_individual != pos_example:
        #                 break
        #
        #         subj_indiv_cls = \
        #             ontology_generator.get_random_range_class(obj_prop)
        #
        #         ontology_generator.add_instance(
        #             subject_individual, subj_indiv_cls)
        #         ontology_generator.add_axiom(
        #             OWLObjectPropertyAssertionAxiom(
        #                 subject_individual, obj_prop, pos_example))
        #
        #     while random() > 0.5:
        #         data_prop = ontology_generator.pick_random_data_property()
        #         datatype = ontology_generator.get_range_datatype(data_prop)
        #         literal = ontology_generator.generate_random_literal(datatype)
        #
        #         ontology_generator.add_axiom(
        #             OWLDataPropertyAssertionAxiom(
        #                 pos_example, data_prop, literal))
        #
        # neg_bridge_individuals = []
        # for neg_example in neg_examples:
        #     while True:
        #         bridge_individual = ontology_generator.pick_random_individual()
        #
        #         if not bridge_individual == neg_example and \
        #                 bridge_individual not in pos_bridge_individuals and \
        #                 not ontology_generator.is_instance_of(
        #                     bridge_individual, target_obj_prop_2_range) and \
        #                 not ontology_generator.is_instance_of(
        #                     bridge_individual, ext_restr_final_filler_class) and \
        #                 bridge_individual not in pos_examples and \
        #                 bridge_individual not in neg_examples:
        #             break
        #
        #     neg_bridge_individuals.append(bridge_individual)
        #
        #     bridge_indiv_cls = choice(
        #         ontology_generator.get_all_sub_classes(target_obj_prop_1_range))
        #     ontology_generator.add_instance(
        #         bridge_individual, bridge_indiv_cls)
        #
        #     ontology_generator.add_axiom(
        #         OWLObjectPropertyAssertionAxiom(
        #             neg_example, target_obj_prop_1, bridge_individual))
        #
        #     while True:
        #         filler_instance = ontology_generator.pick_random_individual()
        #
        #         if not filler_instance == neg_example and \
        #                 not filler_instance == bridge_individual and \
        #                 not ontology_generator.is_instance_of(
        #                     filler_instance, target_obj_prop_1_range) and \
        #                 not ontology_generator.is_instance_of(
        #                     filler_instance, ext_restr_final_filler_class) and \
        #                 not ontology_generator.is_instance_of(
        #                     filler_instance, target_obj_prop_2_pos_range):
        #             break
        #
        #     filler_instance_cls = choice(
        #         ontology_generator.get_all_sub_classes(
        #             target_obj_prop_2_neg_range))
        #     ontology_generator.add_instance(
        #         filler_instance, filler_instance_cls)
        #
        #     ontology_generator.add_axiom(
        #         OWLObjectPropertyAssertionAxiom(
        #             bridge_individual, target_obj_prop_2, filler_instance))
        #
        #     while random() > 0.5:
        #         obj_prop = ontology_generator.pick_random_object_property()
        #
        #         # if obj_prop == target_obj_prop_1 or \
        #         #         obj_prop == target_obj_prop_2:
        #         #     continue
        #
        #         while True:
        #             object_individual = \
        #                 ontology_generator.pick_random_individual()
        #
        #             if object_individual != neg_example:
        #                 break
        #
        #         while True:
        #             obj_indiv_cls = \
        #                 ontology_generator.get_random_range_class(obj_prop)
        #
        #             is_pos_target_2_range_sub_class = \
        #                 obj_indiv_cls in ontology_generator.get_all_sub_classes(
        #                     target_obj_prop_2_pos_range)
        #
        #             is_neg_target_2_range_sub_class = \
        #                 obj_indiv_cls in ontology_generator.get_all_sub_classes(
        #                     target_obj_prop_2_neg_range)
        #
        #             if not is_pos_target_2_range_sub_class and \
        #                     not is_neg_target_2_range_sub_class:
        #                 break
        #
        #         ontology_generator.add_instance(
        #             object_individual, obj_indiv_cls)
        #         ontology_generator.add_axiom(
        #             OWLObjectPropertyAssertionAxiom(
        #                 neg_example, obj_prop, object_individual))
        #
        #     while random() > 0.5:
        #         obj_prop = ontology_generator.pick_random_object_property()
        #
        #         if obj_prop == target_obj_prop_1 or \
        #                 obj_prop == target_obj_prop_2:
        #             continue
        #
        #         while True:
        #             subject_individual = \
        #                 ontology_generator.pick_random_individual()
        #
        #             if subject_individual != neg_example:
        #                 break
        #
        #         subj_indiv_cls = \
        #             ontology_generator.get_random_range_class(obj_prop)
        #
        #         ontology_generator.add_instance(
        #             subject_individual, subj_indiv_cls)
        #         ontology_generator.add_axiom(
        #             OWLObjectPropertyAssertionAxiom(
        #                 subject_individual, obj_prop, neg_example))
        #
        #     while random() > 0.5:
        #         data_prop = ontology_generator.pick_random_data_property()
        #         datatype = ontology_generator.get_range_datatype(data_prop)
        #         literal = ontology_generator.generate_random_literal(datatype)
        #
        #         ontology_generator.add_axiom(
        #             OWLDataPropertyAssertionAxiom(
        #                 neg_example, data_prop, literal))
        #
        # for indiv in non_example_individuals:
        #     if indiv in pos_bridge_individuals or \
        #             indiv in neg_bridge_individuals:
        #         continue
        #
        #     while random() > 0.5:
        #         obj_prop = ontology_generator.pick_random_object_property()
        #
        #         # if obj_prop == target_obj_prop_1 or \
        #         #         obj_prop == target_obj_prop_2:
        #         #     continue
        #
        #         while True:
        #             object_individual = \
        #                 ontology_generator.pick_random_individual()
        #
        #             if object_individual != indiv and \
        #                     object_individual not in pos_bridge_individuals and \
        #                     object_individual not in neg_bridge_individuals:
        #                 break
        #
        #         while True:
        #             obj_indiv_cls = \
        #                 ontology_generator.get_random_range_class(obj_prop)
        #
        #             # is_pos_target_2_range_sub_class = \
        #             #     obj_indiv_cls in ontology_generator.get_all_sub_classes(
        #             #         target_obj_prop_2_pos_range)
        #             #
        #             # is_neg_target_2_range_sub_class = \
        #             #     obj_indiv_cls in ontology_generator.get_all_sub_classes(
        #             #         target_obj_prop_2_neg_range)
        #
        #             # if not is_pos_target_2_range_sub_class and \
        #             #         not is_neg_target_2_range_sub_class:
        #             if True:
        #                 break
        #
        #         ontology_generator.add_instance(
        #             object_individual, obj_indiv_cls)
        #         ontology_generator.add_axiom(
        #             OWLObjectPropertyAssertionAxiom(
        #                 indiv, obj_prop, object_individual))
        #
        #     while random() > 0.5:
        #         obj_prop = ontology_generator.pick_random_object_property()
        #
        #         if obj_prop == target_obj_prop_1 or \
        #                 obj_prop == target_obj_prop_2:
        #             continue
        #
        #         while True:
        #             subject_individual = \
        #                 ontology_generator.pick_random_individual()
        #
        #             if subject_individual != indiv and \
        #                     subject_individual not in pos_bridge_individuals and \
        #                     subject_individual not in neg_bridge_individuals:
        #                 break
        #
        #         subj_indiv_cls = \
        #             ontology_generator.get_random_range_class(obj_prop)
        #
        #         ontology_generator.add_instance(
        #             subject_individual, subj_indiv_cls)
        #         ontology_generator.add_axiom(
        #             OWLObjectPropertyAssertionAxiom(
        #                 subject_individual, obj_prop, indiv))
        #
        #     while random() > 0.5:
        #         data_prop = ontology_generator.pick_random_data_property()
        #         datatype = ontology_generator.get_range_datatype(data_prop)
        #         literal = ontology_generator.generate_random_literal(datatype)
        #
        #         ontology_generator.add_axiom(
        #             OWLDataPropertyAssertionAxiom(
        #                 indiv, data_prop, literal))

        ont = ontology_generator.get_ontology()
        # g = ont.as_rdf_graph()
        # g.serialize(open('/tmp/dbg.ttl', 'wb'), format='turtle')

        print(f'pos: {target_cls_expression}')
        print(f'neg: {target_neg_cls_expression}')

        return LearningScenario(
            {p for p in pos_examples},
            {n for n in neg_examples},
            target_cls_expression,
            ont)


if __name__ == '__main__':
    generator = SimulatedAnnealingLearningScenario1Generator(
        num_classes=50,
        num_data_properties=5,
        num_object_properties=10,
        num_overall_individuals=300,
        num_pos_examples=20,
        num_neg_examples=20,
        existential_nesting_depth=5)

    scenario = generator.generate_scenario()
    scenario.write_sml_bench_scenario('/tmp/sml_bench', 'deleteme')
