from clonal_selection.ackley_tests import test_ackley
from clonal_selection.rastrigin_tests import test_rastrigin
from clonal_selection.schafferF6_tests import test_schafferF6
from clonal_selection.schafferF7_tests import test_schafferF7
from clonal_selection.schwefel_tests import test_schwefel

if __name__ == '__main__':
    print("---------- Schwefel Problem ----------")
    test_schwefel()
    print("---------- Schaffer F7 Problem ----------")
    test_schafferF7()
    print("---------- Schaffer F6 Problem ----------")
    test_schafferF6()
    print("---------- Rastrigin Problem ----------")
    test_rastrigin()
    print("---------- Ackley Problem ----------")
    test_ackley()