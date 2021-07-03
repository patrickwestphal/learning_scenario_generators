import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import randint, gauss, choice, shuffle
from typing import Set, List

from lorem import get_word
from morelianoctua.model import OWLOntology, OWLAxiom
from morelianoctua.model.axioms.assertionaxiom import \
    OWLDataPropertyAssertionAxiom, OWLClassAssertionAxiom
from morelianoctua.model.axioms.classaxiom import OWLSubClassOfAxiom
from morelianoctua.model.axioms.declarationaxiom import \
    OWLClassDeclarationAxiom, OWLDataPropertyDeclarationAxiom, \
    OWLNamedIndividualDeclarationAxiom, OWLObjectPropertyDeclarationAxiom
from morelianoctua.model.axioms.owldatapropertyaxiom import \
    OWLDataPropertyDomainAxiom, OWLDataPropertyRangeAxiom
from morelianoctua.model.axioms.owlobjectpropertyaxiom import \
    OWLObjectPropertyDomainAxiom, OWLObjectPropertyRangeAxiom
from morelianoctua.model.objects.classexpression import OWLClassExpression, \
    OWLClass
from morelianoctua.model.objects.datarange import OWLDatatype
from morelianoctua.model.objects.individual import OWLNamedIndividual
from morelianoctua.model.objects.property import OWLDataProperty, \
    OWLObjectProperty
from morelianoctua.reasoning.owllinkreasoner import OWLLinkReasoner
from rdflib import URIRef, Literal, XSD, OWL


@dataclass
class LearningScenario:
    positive_examples: Set[OWLNamedIndividual]
    negative_examples: Set[OWLNamedIndividual]
    target_concept: OWLClassExpression
    background_knowledge: OWLOntology

    def write_sml_bench_scenario(self, base_path: str, learning_task_name: str):
        learning_task_dir_path = os.path.join(base_path, learning_task_name)
        os.mkdir(learning_task_dir_path)

        lt_owl_dir_path = os.path.join(learning_task_dir_path, 'owl')
        os.mkdir(lt_owl_dir_path)

        data_dir = os.path.join(lt_owl_dir_path, 'data')
        os.mkdir(data_dir)

        data_file_path = os.path.join(data_dir, learning_task_name + '.owl')
        with open(data_file_path, 'wb') as data_file:
            kg = self.background_knowledge.as_rdf_graph()
            kg.serialize(data_file, format='xml')

        lps_dir = os.path.join(lt_owl_dir_path, 'lp')
        os.mkdir(lps_dir)
        lp_dir = os.path.join(lps_dir, '1')
        os.mkdir(lp_dir)

        pos_file_path = os.path.join(lp_dir, 'pos.txt')
        with open(pos_file_path, 'w') as pos_file:
            for pos_example in self.positive_examples:
                pos_file.write(str(pos_example.iri) + os.linesep)

        neg_file_path = os.path.join(lp_dir, 'neg.txt')
        with open(neg_file_path, 'w') as neg_file:
            for neg_example in self.negative_examples:
                neg_file.write(str(neg_example.iri) + os.linesep)

        info_file_path = os.path.join(lp_dir, 'info.txt')
        with open(info_file_path, 'w') as info_file:
            info_file.write(str(self.target_concept))

        # import pdb; pdb.set_trace()
        # pass


class OntologyGenerator:
    def __init__(self):

        self.classes = []
        self.object_properties = []
        self.data_properties = []
        self.individuals = []

        self.axioms = set()

        self.class_hierarchy = {}

        self._possible_data_property_ranges = [
            OWLDatatype(XSD.int),
            OWLDatatype(XSD.double),
            OWLDatatype(XSD.string)]

        self._object_property_domains = {}
        self._object_property_ranges = {}
        self._data_property_domains = {}
        self._data_property_ranges = {}

        self._individuals_types = {}

        self._cls_cntr = 0
        self._obj_prop_cntr = 0
        self._data_prop_cntr = 0
        self._indiv_cntr = 0
        self._literal_cntr = 0

        self.ontology_prefix = \
            'http://dl-learner.org/ontology%06i#' % randint(1, 999999)

    def _next_cls_iri(self):
        self._cls_cntr += 1
        return URIRef(self.ontology_prefix + f'Cls{self._cls_cntr}')

    def _next_obj_prop_iri(self):
        self._obj_prop_cntr += 1
        return URIRef(self.ontology_prefix + f'objProp{self._obj_prop_cntr}')

    def _next_data_prop_iri(self):
        self._data_prop_cntr += 1
        return URIRef(self.ontology_prefix + f'dataProp{self._data_prop_cntr}')

    def add_new_class(self):
        cls = OWLClass(self._next_cls_iri())
        self.classes.append(cls)
        self.axioms.add(OWLClassDeclarationAxiom(cls))

    def add_new_object_property(self):
        obj_prop = OWLObjectProperty(self._next_obj_prop_iri())
        self.object_properties.append(obj_prop)
        self.axioms.add(OWLObjectPropertyDeclarationAxiom(obj_prop))

    def add_new_data_property(self):
        data_prop = OWLDataProperty(self._next_data_prop_iri())
        self.data_properties.append(data_prop)
        self.axioms.add(OWLDataPropertyDeclarationAxiom(data_prop))

    def add_new_individual(self, local_part_prefix: str = 'indiv'):
        self._indiv_cntr += 1
        indiv = OWLNamedIndividual(
            self.ontology_prefix + local_part_prefix + str(self._indiv_cntr))

        self.individuals.append(indiv)
        self.axioms.add(OWLNamedIndividualDeclarationAxiom(indiv))

        return indiv

    def add_axiom(self, axiom: OWLAxiom):
        self.axioms.add(axiom)

    def init_random_class_hierarchy(self):
        owl_thing = OWLClass(OWL.Thing)
        super_classes = [owl_thing]
        self.class_hierarchy = {owl_thing: []}

        for cls in self.classes:
            self.axioms.add(OWLClassDeclarationAxiom(cls))
            super_class = choice(super_classes)

            self.axioms.add(OWLSubClassOfAxiom(cls, super_class))

            if self.class_hierarchy.get(cls) is None:
                self.class_hierarchy[cls] = []

            self.class_hierarchy[super_class].append(cls)
            super_classes.append(cls)

    def get_all_sub_classes(self, super_cls: OWLClass) -> List[OWLClass]:
        sub_classes = self.class_hierarchy[super_cls].copy()
        sub_classes.append(super_cls)

        for sub_cls in self.class_hierarchy[super_cls]:
            sub_classes += self.get_all_sub_classes(sub_cls)

        return sub_classes

    def get_all_direct_sub_classes(self, super_cls: OWLClass) -> List[OWLClass]:
        return self.class_hierarchy[super_cls].copy()

    def get_classes_from_complement_of_subtree(
            self, cls: OWLClass) -> List[OWLClass]:

        classes_to_ignore = self.get_all_sub_classes(cls)

        return [c for c in set(self.classes).difference(classes_to_ignore)]

    def pick_random_class(self):
        return choice(self.classes)

    def set_object_property_domain_and_range(
            self,
            object_property: OWLObjectProperty,
            domain_cls: OWLClass,
            range_cls: OWLClass):
        self._object_property_domains[object_property] = domain_cls
        self.axioms.add(
            OWLObjectPropertyDomainAxiom(object_property, domain_cls))

        self._object_property_ranges[object_property] = range_cls
        self.axioms.add(OWLObjectPropertyRangeAxiom(object_property, range_cls))

    def set_data_property_domain_and_range(
            self,
            data_property: OWLDataProperty,
            domain_cls: OWLClass,
            range_dtype: OWLDatatype):
        self._data_property_domains[data_property] = domain_cls
        self.axioms.add(OWLDataPropertyDomainAxiom(data_property, domain_cls))

        self._data_property_ranges[data_property] = range_dtype
        self.axioms.add(OWLDataPropertyRangeAxiom(data_property, range_dtype))

    def pick_random_object_property(self) -> OWLObjectProperty:
        return choice(self.object_properties)

    def pick_random_data_property(self) -> OWLDataProperty:
        return choice(self.data_properties)

    def pick_random_individual(self) -> OWLNamedIndividual:
        return choice(self.individuals)

    def pick_random_individual_by_cls(
            self, cls: OWLClass) -> OWLNamedIndividual:

        indivs = self.individuals.copy()
        shuffle(indivs)

        for indiv in self.individuals:
            if self.is_instance_of(indiv, cls):
                return indiv

        return None

    def pick_random_datatype(self) -> OWLDatatype:
        return choice(self._possible_data_property_ranges)

    def get_ontology(self) -> OWLOntology:
        return OWLOntology(prefix_declarations={}, axioms=self.axioms)

    def pick_random_cls_w_at_least_two_sub_classes(self):
        # a range class is to be found that has at least one sibling
        # class which then would become the target class of the negative
        # examples
        cls_candidates = self.classes.copy()
        shuffle(cls_candidates)
        cls = None
        for cls_candidate in cls_candidates:
            if len(self.class_hierarchy[cls_candidate]) > 1:
                cls = cls_candidate
                break

        if cls is None:
            raise Exception(
                'Generated class hierarchy doesn\'t allow the creation '
                'of a proper learning problem. Please re-run.')

        return cls

    def add_instance(self, instance: OWLNamedIndividual, cls: OWLClass):
        self.axioms.add(OWLClassAssertionAxiom(instance, cls))
        if self._individuals_types.get(instance) is None:
            self._individuals_types[instance] = []

        self._individuals_types[instance].append(cls)

    def has_types(self, individual: OWLNamedIndividual):
        if self._individuals_types.get(individual) is None:
            return []
        else:
            return self._individuals_types[individual]

    is_instance_call_cntr = 0
    def is_instance_of(self, individual: OWLNamedIndividual, cls: OWLClass) \
            -> bool:

        reasoner = OWLLinkReasoner(self.get_ontology(), 'http://localhost:8383')
        # time.sleep(1)
        is_instance = reasoner.is_entailed(
            OWLClassAssertionAxiom(individual, cls))

        reasoner.release_kb()

        self.is_instance_call_cntr += 1
        if self.is_instance_call_cntr == 15000:
            self.is_instance_call_cntr = 0
            print('Restart OWLlink server')
            input('')
        return is_instance

    def get_random_range_class(self, object_property: OWLObjectProperty):
        rnd_range_cls = choice(
            self.get_all_sub_classes(
                self._object_property_ranges[object_property]))

        return rnd_range_cls

    def get_random_domain_class(self, object_property: OWLObjectProperty):
        rnd_dom_cls = choice(
            self.get_all_sub_classes(
                self._object_property_domains[object_property]))

        return rnd_dom_cls

    def get_object_properties_by_domain(
            self, domain: OWLClass) -> List[OWLObjectProperty]:

        obj_props = []

        for _op in self.object_properties:
            domain_classes = self.get_all_sub_classes(
                self._object_property_domains[_op])

            if domain in domain_classes:
                obj_props.append(_op)

        return obj_props

    def get_object_properties_by_range(
            self, rnge: OWLClass) -> List[OWLObjectProperty]:

        obj_props = []

        for _op in self.object_properties:
            range_classes = self.get_all_sub_classes(
                self._object_property_ranges[_op])

            if rnge in range_classes:
                obj_props.append(_op)

        return obj_props

    def get_range_cls(self, object_property: OWLObjectProperty) -> OWLClass:
        return self._object_property_ranges[object_property]

    def get_domain_cls(self, object_property: OWLObjectProperty) -> OWLClass:
        return self._object_property_domains[object_property]

    def get_range_datatype(self, data_property: OWLDataProperty) -> OWLDatatype:
        return self._data_property_ranges[data_property]

    @staticmethod
    def generate_random_literal(datatype: OWLDatatype):
        if datatype.iri == XSD.string:
            return Literal(get_word(), None, XSD.string)
        elif datatype.iri == XSD.int:
            return Literal(randint(1, 23), None, XSD.int)
        elif datatype.iri == XSD.double:
            return Literal(gauss(23, 5), None, XSD.double)
        else:
            raise Exception(f'Unhandled data type {str(datatype)}')


class LearningScenarioGenerator(ABC):
    @abstractmethod
    def generate_scenario(self) -> LearningScenario:
        raise NotImplementedError()
