
from dataclasses import dataclass
from bakery import assert_equal
from drafter import *
import re
import math


set_website_title("Chemical Reaction Calculator")
set_site_information(
    author="Charles Tisdale",
    description=(
        "A student chemistry calculator for balanced reactions, "
        "limiting reactants, yields, and thermochemistry."
    ),
    sources=(
        "Atomic masses from course reference values; "
        "Drafter documentation; AI-assisted development "
        "see AI conversation transcript in the repository"
    ),
    planning=["planning.pdf, designing.pdf"],
    links=["https://github.com/charlesctisdale/chem-calculator"]
)



# ==========================================
# DATA CLASSES
# ==========================================

@dataclass
class Reactant:
    name: str
    mass: float
    molar_mass: float
    coefficient: int


@dataclass
class State:
    equation: str
    reactants: list[Reactant]
    product: str
    actual_yield: float
    reaction_energy: float
    limiting_reactant: str
    result: str


def make_initial_state() -> State:
    """Create a fresh state with no reaction information."""
    return State("", [], "", 0.0, 0.0, "", "")


def test_make_initial_state() -> None:
    """Test that all state fields begin empty or at zero."""
    state = make_initial_state()
    assert_equal(state.equation, "")
    assert_equal(state.reactants, [])
    assert_equal(state.product, "")
    assert_equal(state.actual_yield, 0.0)
    assert_equal(state.reaction_energy, 0.0)
    assert_equal(state.limiting_reactant, "")
    assert_equal(state.result, "")



# ==========================================
# ATOMIC MASSES
# ==========================================

ATOMIC_MASSES = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.94,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.01,
    "N": 14.01,
    "O": 16.00,
    "F": 19.00,
    "Na": 22.99,
    "Mg": 24.31,
    "Al": 26.98,
    "Si": 28.09,
    "P": 30.97,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.10,
    "Ca": 40.08,
    "Fe": 55.85,
    "Cu": 63.55,
    "Zn": 65.38,
    "Ag": 107.87,
    "Ba": 137.33
}


# ==========================================
# CHEMISTRY FUNCTIONS
# ==========================================

def count_atoms(formula: str) -> dict[str, int]:
    """Count elements in a simple formula without parentheses or charges."""
    atoms: dict[str, int] = {}
    position = 0

    if formula == "":
        raise ValueError("Empty chemical formula.")

    # Regular expression: [A-Z][a-z]? matches an element symbol,
    # and [0-9]* matches an optional atom count, e.g. H2 or Na.
    # The position check below rejects unsupported symbols like parentheses.
    pattern = r"([A-Z][a-z]?)([0-9]*)"

    for match in re.finditer(pattern, formula):
        if match.start() != position:
            raise ValueError(
                "Unsupported formula: " + formula
            )

        element = match.group(1)
        number = match.group(2)

        if element not in ATOMIC_MASSES:
            raise ValueError(
                "Unknown element: " + element
            )

        count = int(number) if number else 1

        if count <= 0:
            raise ValueError("Invalid atom count.")

        atoms[element] = (
            atoms.get(element, 0) + count
        )

        position = match.end()

    if position != len(formula):
        raise ValueError(
            "Unsupported formula: " + formula
        )

    return atoms


def test_count_atoms() -> None:
    """Test counting elements and repeated element symbols."""
    assert_equal(count_atoms("H2O"), {"H": 2, "O": 1})
    assert_equal(count_atoms("Al2O3"), {"Al": 2, "O": 3})
    assert_equal(count_atoms("CH3CH3"), {"C": 2, "H": 6})
    try:
        count_atoms("Ca(OH)2")
    except ValueError:
        error_found = True
    else:
        error_found = False
    assert_equal(error_found, True)



def calculate_molar_mass(formula: str) -> float:
    """Sum the atomic masses for all atoms in a chemical formula."""
    atoms = count_atoms(formula)
    total = 0.0

    for element, count in atoms.items():
        total += ATOMIC_MASSES[element] * count

    return total


def test_calculate_molar_mass() -> None:
    """Test molar mass calculations with known course values."""
    assert_equal(round(calculate_molar_mass("H2O"), 3), 18.016)
    assert_equal(round(calculate_molar_mass("CO2"), 2), 44.01)



def get_species(side: str) -> list[tuple[int, str]]:
    """Parse a reaction side into coefficient and formula pairs."""
    species: list[tuple[int, str]] = []

    for item in side.split("+"):
        item = item.strip()

        # This pattern separates a leading reaction coefficient from
        # its formula, for example "2H2O" becomes (2, "H2O").
        match = re.fullmatch(
            r"([0-9]*)\s*([A-Z][A-Za-z0-9]*)",
            item
        )

        if match is None:
            raise ValueError(
                "Invalid equation format."
            )

        coefficient_text = match.group(1)
        formula = match.group(2)

        if coefficient_text == "":
            coefficient = 1
        else:
            coefficient = int(coefficient_text)

        if coefficient <= 0:
            raise ValueError(
                "Coefficients must be positive."
            )

        count_atoms(formula)

        species.append((coefficient, formula))

    return species


def test_get_species() -> None:
    """Test extracting formulas and implied coefficients of one."""
    assert_equal(get_species("2H2 + O2"), [(2, "H2"), (1, "O2")])
    assert_equal(get_species(" 2H2O "), [(2, "H2O")])



def split_equation(
    equation: str
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Separate a reaction equation into its left and right species."""
    parts = equation.split("->")

    if len(parts) != 2:
        raise ValueError(
            "Use -> between reactants and products."
        )

    reactants = get_species(parts[0])
    products = get_species(parts[1])

    return reactants, products


def test_split_equation() -> None:
    """Test separating reactants and products."""
    expected = ([(2, "H2"), (1, "O2")], [(2, "H2O")])
    assert_equal(split_equation("2H2 + O2 -> 2H2O"), expected)



def is_balanced(equation: str) -> bool:
    """Check whether both sides contain equal counts of each atom."""
    reactants, products = split_equation(equation)

    left_atoms: dict[str, int] = {}
    right_atoms: dict[str, int] = {}

    for coefficient, formula in reactants:
        atoms = count_atoms(formula)

        for element, count in atoms.items():
            left_atoms[element] = (
                left_atoms.get(element, 0)
                + count * coefficient
            )

    for coefficient, formula in products:
        atoms = count_atoms(formula)

        for element, count in atoms.items():
            right_atoms[element] = (
                right_atoms.get(element, 0)
                + count * coefficient
            )

    return left_atoms == right_atoms


def test_is_balanced() -> None:
    """Test balanced and unbalanced equations."""
    assert_equal(is_balanced("2H2 + O2 -> 2H2O"), True)
    assert_equal(is_balanced("H2 + O2 -> H2O"), False)



def find_coefficient(
    species: list[tuple[int, str]], formula: str
) -> int:
    """Find a product or reactant coefficient by chemical formula."""
    for coefficient, name in species:
        if name == formula:
            return coefficient

    raise ValueError(
        "Chemical not found: " + formula
    )


def test_find_coefficient() -> None:
    """Test looking up an individual chemical's coefficient."""
    species = [(2, "H2"), (1, "O2")]
    assert_equal(find_coefficient(species, "H2"), 2)
    assert_equal(find_coefficient(species, "O2"), 1)



# ==========================================
# MAIN CALCULATIONS
# ==========================================

def calculate_chemistry(state: State) -> str:

    """Compute limiting reactant, yields, and leftovers from state."""
    reactant_species, product_species = split_equation(
        state.equation
    )

    if not is_balanced(state.equation):
        raise ValueError(
            "Your equation is not balanced."
        )

    if len(state.reactants) != len(reactant_species):
        raise ValueError(
            "Not all reactant masses were entered."
        )

    if any(
        not math.isfinite(r.mass) or r.mass < 0
        or r.molar_mass <= 0 or r.coefficient <= 0
        for r in state.reactants
    ):
        raise ValueError("Reactant values must be valid and nonnegative.")

    # Find limiting reactant
    # Infinity guarantees the first valid ratio is the starting minimum.
    # This identifies the reactant producing the least reaction progress.
    smallest_ratio = float("inf")
    limiting = ""

    # Each Reactant is an instance of our second dataclass. This loop
    # is the assignment's required meaningful iteration over State.reactants.
    for reactant in state.reactants:
        moles = (
            reactant.mass / reactant.molar_mass
        )

        ratio = (
            moles / reactant.coefficient
        )

        if ratio < smallest_ratio:
            smallest_ratio = ratio
            limiting = reactant.name

    state.limiting_reactant = limiting

    # Calculate product moles
    product_coefficient = find_coefficient(
        product_species,
        state.product
    )

    product_moles = (
        smallest_ratio * product_coefficient
    )

    product_molar_mass = calculate_molar_mass(
        state.product
    )

    # Calculate theoretical yield
    theoretical_yield = (
        product_moles * product_molar_mass
    )

    if not math.isfinite(state.actual_yield) or state.actual_yield < 0:
        raise ValueError("Actual yield must be valid and nonnegative.")

    # Calculate percent yield
    if theoretical_yield > 0:
        percent_yield = (
            state.actual_yield
            / theoretical_yield
            * 100
        )
    else:
        percent_yield = None

    # Make results text
    output = (
        "Limiting Reactant: "
        + limiting + "\n"
    )

    output += (
        "Product: "
        + state.product + "\n"
    )

    output += (
        "Theoretical Yield: "
        + str(round(theoretical_yield, 3))
        + " g\n"
    )

    output += (
        "Actual Yield: "
        + str(round(state.actual_yield, 3))
        + " g\n"
    )

    if percent_yield is None:
        output += (
            "Percent Yield: Undefined "
            "(theoretical yield is zero)\n"
        )
    else:
        output += (
            "Percent Yield: "
            + str(round(percent_yield, 2))
            + "%\n"
        )

    output += "\nRemaining Reactants:\n"

    for reactant in state.reactants:

        initial_moles = (
            reactant.mass / reactant.molar_mass
        )

        used_moles = (
            smallest_ratio * reactant.coefficient
        )

        remaining_moles = max(
            0.0,
            initial_moles - used_moles
        )

        remaining_mass = (
            remaining_moles * reactant.molar_mass
        )

        output += (
            reactant.name + ": "
            + str(round(remaining_mass, 3))
            + " g\n"
        )

    return output


def test_calculate_chemistry() -> None:
    """Test limiting reactant, yields, and leftovers together."""
    state = make_initial_state()
    state.equation = "2H2 + O2 -> 2H2O"
    state.reactants = [
        Reactant("H2", 4.032, 2.016, 2),
        Reactant("O2", 48.0, 32.0, 1)
    ]
    state.product = "H2O"
    state.actual_yield = 30.0
    result = calculate_chemistry(state)
    assert_equal(state.limiting_reactant, "H2")
    assert_equal("Theoretical Yield: 36.032 g" in result, True)
    assert_equal("Percent Yield: 83.26%" in result, True)
    assert_equal("O2: 16.0 g" in result, True)



# ==========================================
# PAGE 1: HOME PAGE
# ==========================================

@route
def index(state: State) -> Page:
    """Display the home page and the main navigation choices."""
    return Page(state, [
        Header("Chemical Reaction Calculator"),

        Text(
            "Welcome to the Chemical Reaction "
            "Calculator!"
        ),

        Text(
            "This website helps you calculate "
            "the results of a chemical reaction."
        ),

        Text("Features include:"),

        Text("- Checking balanced equations"),
        Text("- Calculating molar masses"),
        Text("- Finding limiting reactants"),
        Text("- Calculating theoretical yield"),
        Text("- Calculating percent yield"),
        Text("- Finding leftover reactants"),
        Text("- Classifying reaction energy"),

        Button(
            "Start Calculator",
            reaction_setup
        ),

        Button(
            "Learn More",
            learn_more
        )
    ])


def test_index() -> None:
    """Smoke-test the homepage; browser layout needs manual testing."""
    page = index(make_initial_state())
    assert_equal(page is not None, True)



# ==========================================
# PAGE 2: CHEMICAL EQUATION
# ==========================================

@route
def reaction_setup(state: State) -> Page:
    """Display the balanced chemical-equation entry page."""
    return Page(state, [
        Header("Enter Your Chemical Equation"),

        Text(
            "Enter the balanced chemical "
            "equation for your reaction."
        ),

        Text(
            "Use -> to separate reactants "
            "and products."
        ),

        Text("Example:"),

        Text("2H2 + O2 -> 2H2O"),

        TextBox("equation", state.equation),

        Button(
            "Continue",
            save_equation
        ),

        Button(
            "Back Home",
            index
        )
    ])


def test_reaction_setup() -> None:
    """Smoke-test equation entry; textbox submission needs a browser."""
    page = reaction_setup(make_initial_state())
    assert_equal(page is not None, True)



# ==========================================
# SAVE EQUATION
# ==========================================

@route
def save_equation(
    state: State,
    equation: str
) -> Page:

    """Validate the equation, reset previous inputs, and continue."""
    try:
        equation = equation.strip()

        if not is_balanced(equation):
            return Page(state, [
                Header("Equation Error"),

                Text(
                    "Your chemical equation "
                    "is not balanced."
                ),

                Text(
                    "Please check the coefficients "
                    "and try again."
                ),

                Button(
                    "Try Again",
                    reaction_setup
                )
            ])

        reactants, products = split_equation(equation)

        reactant_names = [
            formula for coefficient, formula
            in reactants
        ]

        product_names = [formula for coefficient, formula in products]

        if len(set(product_names)) != len(product_names):
            raise ValueError("Combine repeated products first.")

        if len(set(reactant_names)) != len(reactant_names):
            raise ValueError(
                "Combine repeated reactants first."
            )

        state.equation = equation
        state.reactants = []
        state.product = ""
        state.actual_yield = 0.0
        state.reaction_energy = 0.0
        state.limiting_reactant = ""
        state.result = ""

        return reactant_input(state)

    except ValueError as error:
        return Page(state, [
            Header("Equation Error"),

            Text(str(error)),

            Button(
                "Try Again",
                reaction_setup
            )
        ])


def test_save_equation() -> None:
    """Test valid equation storage and rejection of unbalanced input."""
    state = make_initial_state()
    state.result = "old"
    page = save_equation(state, " 2H2 + O2 -> 2H2O ")
    assert_equal(page is not None, True)
    assert_equal(state.equation, "2H2 + O2 -> 2H2O")
    assert_equal(state.result, "")
    invalid = make_initial_state()
    save_equation(invalid, "H2 + O2 -> H2O")
    assert_equal(invalid.equation, "")



# ==========================================
# PAGE 3: ENTER REACTANT MASSES
# ==========================================

@route
def reactant_input(state: State) -> Page:

    """Show the input page for the next unrecorded reactant."""
    reactants, products = split_equation(
        state.equation
    )

    current = len(state.reactants)

    # Move to product page when finished
    if current >= len(reactants):
        return product_input(state)

    coefficient, formula = reactants[current]

    return Page(state, [
        Header("Enter Reactant Masses"),

        Text("Your chemical equation:"),

        Text(state.equation),

        Text(
            "Reactant "
            + str(current + 1)
            + " of "
            + str(len(reactants))
        ),

        Header(formula),

        Text(
            "How many grams of "
            + formula
            + " are you starting with?"
        ),

        Text(
            "Enter only the mass as a number."
        ),

        Text("Example: 10.5"),

        TextBox("mass", "", "number"),

        Button(
            "Save and Continue",
            save_reactant_mass
        ),

        Button(
            "Start Over",
            clear_reaction
        )
    ])


def test_reactant_input() -> None:
    """Smoke-test reactant entry; rendered inputs need browser testing."""
    state = make_initial_state()
    state.equation = "2H2 + O2 -> 2H2O"
    page = reactant_input(state)
    assert_equal(page is not None, True)
    assert_equal(len(state.reactants), 0)



# ==========================================
# SAVE REACTANT MASS
# ==========================================

@route
def save_reactant_mass(
    state: State,
    mass: str
) -> Page:

    """Validate and store one reactant entry, then advance."""
    try:
        mass_value = float(mass)

        if not math.isfinite(mass_value) or mass_value < 0:
            raise ValueError(
                "Mass cannot be negative."
            )

        reactants, products = split_equation(
            state.equation
        )

        current = len(state.reactants)

        if current >= len(reactants):
            return product_input(state)

        coefficient, formula = reactants[current]

        new_reactant = Reactant(
            formula,
            mass_value,
            calculate_molar_mass(formula),
            coefficient
        )

        state.reactants.append(new_reactant)

        return reactant_input(state)

    except ValueError:
        return Page(state, [
            Header("Invalid Mass"),

            Text(
                "Please enter a valid number "
                "greater than or equal to zero."
            ),

            Button(
                "Try Again",
                reactant_input
            )
        ])


def test_save_reactant_mass() -> None:
    """Test saving reactants and rejecting negative input."""
    state = make_initial_state()
    state.equation = "2H2 + O2 -> 2H2O"
    save_reactant_mass(state, "4.032")
    assert_equal(len(state.reactants), 1)
    assert_equal(state.reactants[0].name, "H2")
    assert_equal(state.reactants[0].coefficient, 2)
    save_reactant_mass(state, "-5")
    assert_equal(len(state.reactants), 1)
    save_reactant_mass(state, "48")
    assert_equal(len(state.reactants), 2)



# ==========================================
# PAGE 4: PRODUCT INPUT
# ==========================================

@route
def product_input(state: State) -> Page:

    """Ask for the actual yield of the first listed product."""
    reactants, products = split_equation(
        state.equation
    )

    product_formula = products[0][1]

    return Page(state, [
        Header("Product Information"),

        Text("Your chemical equation:"),

        Text(state.equation),

        Text("Product being calculated:"),

        Header(product_formula),

        Text(
            "How many grams of "
            + product_formula
            + " did you actually collect?"
        ),

        Text(
            "This is called the actual yield."
        ),

        Text(
            "If you only want theoretical yield, "
            "enter 0 (percent yield will show 0%)."
        ),

        Text("Example: 15.5"),

        TextBox(
            "actual_mass",
            "",
            "number"
        ),

        Button(
            "Calculate Results",
            finish_calculation
        ),

        Button(
            "Start Over",
            clear_reaction
        )
    ])


def test_product_input() -> None:
    """Smoke-test product page; textbox input needs browser testing."""
    state = make_initial_state()
    state.equation = "2H2 + O2 -> 2H2O"
    page = product_input(state)
    assert_equal(page is not None, True)



# ==========================================
# FINISH CALCULATION
# ==========================================

@route
def finish_calculation(
    state: State,
    actual_mass: str
) -> Page:

    """Validate actual yield, calculate, and open results."""
    try:
        reactants, products = split_equation(
            state.equation
        )

        state.product = products[0][1]

        state.actual_yield = float(
            actual_mass
        )

        if (not math.isfinite(state.actual_yield)
                or state.actual_yield < 0):
            raise ValueError(
                "Actual yield cannot be negative."
            )

        state.result = calculate_chemistry(
            state
        )

        return results(state)

    except ValueError as error:
        return Page(state, [
            Header("Calculation Error"),

            Text(str(error)),

            Button(
                "Try Again",
                product_input
            )
        ])


def test_finish_calculation() -> None:
    """Test route saving actual yield and storing calculation text."""
    state = make_initial_state()
    state.equation = "2H2 + O2 -> 2H2O"
    state.reactants = [
        Reactant("H2", 4.032, 2.016, 2),
        Reactant("O2", 48.0, 32.0, 1)
    ]
    page = finish_calculation(state, "30")
    assert_equal(page is not None, True)
    assert_equal(state.actual_yield, 30.0)
    assert_equal(state.product, "H2O")
    assert_equal("Percent Yield: 83.26%" in state.result, True)



# ==========================================
# PAGE 5: RESULTS
# ==========================================

@route
def results(state: State) -> Page:
    """Display stored reaction calculations and next-page links."""
    return Page(state, [
        Header("Reaction Results"),

        Text(
            "Equation: " + state.equation
        ),

        Text(state.result),

        Button(
            "View Reactant Details",
            view_reactants
        ),

        Button(
            "Thermochemistry",
            thermochemistry
        ),

        Button(
            "Start Over",
            clear_reaction
        ),

        Button(
            "Home",
            index
        )
    ])


def test_results() -> None:
    """Smoke-test results page; actual rendering needs browser testing."""
    state = make_initial_state()
    state.result = "example result"
    page = results(state)
    assert_equal(page is not None, True)
    assert_equal(state.result, "example result")



# ==========================================
# PAGE 6: REACTANT DETAILS
# ==========================================

@route
def view_reactants(state: State) -> Page:

    """List the entered reactant properties in a useful summary."""
    details = []

    for reactant in state.reactants:

        details.append(
            Text(
                "Reactant: "
                + reactant.name
            )
        )

        details.append(
            Text(
                "Starting Mass: "
                + str(reactant.mass)
                + " g"
            )
        )

        details.append(
            Text(
                "Molar Mass: "
                + str(round(reactant.molar_mass, 3))
                + " g/mol"
            )
        )

        details.append(
            Text(
                "Coefficient: "
                + str(reactant.coefficient)
            )
        )

    return Page(state, [
        Header("Reactant Details"),

        *details,

        Button(
            "Back to Results",
            results
        )
    ])


def test_view_reactants() -> None:
    """Smoke-test reactant details; visual layout needs browser testing."""
    state = make_initial_state()
    state.reactants = [Reactant("O2", 32.0, 32.0, 1)]
    page = view_reactants(state)
    assert_equal(page is not None, True)



# ==========================================
# PAGE 7: THERMOCHEMISTRY
# ==========================================

@route
def thermochemistry(state: State) -> Page:
    """Request the reaction enthalpy for classification."""
    return Page(state, [
        Header("Thermochemistry"),

        Text(
            "Thermochemistry studies heat "
            "changes during chemical reactions."
        ),

        Text(
            "Enter the reaction enthalpy "
            "(Delta H) in kJ per mole of "
            "reaction as written."
        ),

        Text(
            "Negative Delta H = Exothermic"
        ),

        Text(
            "Positive Delta H = Endothermic"
        ),

        TextBox(
            "energy",
            str(state.reaction_energy),
            "number"
        ),

        Button(
            "Classify Reaction",
            save_energy
        ),

        Button(
            "Back to Results",
            results
        )
    ])


def test_thermochemistry() -> None:
    """Smoke-test enthalpy page; browser interaction needs manual testing."""
    page = thermochemistry(make_initial_state())
    assert_equal(page is not None, True)



# ==========================================
# SAVE REACTION ENERGY
# ==========================================

@route
def save_energy(
    state: State,
    energy: str
) -> Page:

    """Save and classify the signed reaction enthalpy."""
    try:
        state.reaction_energy = float(
            energy
        )

        if not math.isfinite(state.reaction_energy):
            raise ValueError("Energy must be a finite number.")

        if state.reaction_energy < 0:
            classification = "Exothermic"

        elif state.reaction_energy > 0:
            classification = "Endothermic"

        else:
            classification = (
                "Neither (Delta H = 0)"
            )

        return Page(state, [
            Header("Thermochemistry Results"),

            Text(
                "Delta H: "
                + str(state.reaction_energy)
                + " kJ/mol of reaction"
            ),

            Text(
                "Classification: "
                + classification
            ),

            Button(
                "Back to Results",
                results
            )
        ])

    except ValueError:
        return Page(state, [
            Header("Invalid Energy"),

            Text(
                "Please enter a valid finite number."
            ),

            Button(
                "Try Again",
                thermochemistry
            )
        ])


def test_save_energy() -> None:
    """Test saving positive, negative, zero, and invalid energy values."""
    for value in ("-50", "25", "0"):
        state = make_initial_state()
        page = save_energy(state, value)
        assert_equal(page is not None, True)
        assert_equal(state.reaction_energy, float(value))
    invalid = make_initial_state()
    save_energy(invalid, "nan")
    assert_equal(math.isfinite(invalid.reaction_energy), False)



# ==========================================
# PAGE 8: LEARN MORE
# ==========================================

@route
def learn_more(state: State) -> Page:
    """Explain chemistry terms and available calculator features."""
    return Page(state, [
        Header("Learn More About Chemistry"),

        Header("Balanced Equations"),

        Text(
            "A balanced equation has the same "
            "number of atoms of each element "
            "on both sides."
        ),

        Header("Molar Mass"),

        Text(
            "Molar mass is the mass of one "
            "mole of a substance in grams."
        ),

        Header("Limiting Reactant"),

        Text(
            "The limiting reactant is the "
            "reactant that runs out first."
        ),

        Text(
            "It determines the maximum "
            "amount of product that can form."
        ),

        Header("Theoretical Yield"),

        Text(
            "Theoretical yield is the maximum "
            "mass of product that can form "
            "from a chemical reaction."
        ),

        Header("Percent Yield"),

        Text(
            "Percent yield compares the actual "
            "yield to the theoretical yield."
        ),

        Text(
            "Percent Yield = "
            "(Actual / Theoretical) * 100"
        ),

        Header("Exothermic Reactions"),

        Text(
            "Exothermic reactions release "
            "heat and have negative Delta H."
        ),

        Header("Endothermic Reactions"),

        Text(
            "Endothermic reactions absorb "
            "heat and have positive Delta H."
        ),

        Button(
            "Back Home",
            index
        )
    ])


def test_learn_more() -> None:
    """Smoke-test definitions page; actual rendering needs a browser."""
    page = learn_more(make_initial_state())
    assert_equal(page is not None, True)



# ==========================================
# RESET CALCULATOR
# ==========================================

@route
def clear_reaction(state: State) -> Page:

    """Reset all seven state fields and return to the home page."""
    state.equation = ""
    state.reactants = []
    state.product = ""
    state.actual_yield = 0.0
    state.reaction_energy = 0.0
    state.limiting_reactant = ""
    state.result = ""

    return index(state)


def test_clear_reaction() -> None:
    """Test that restarting resets all seven State fields."""
    state = make_initial_state()
    state.equation = "2H2 + O2 -> 2H2O"
    state.reactants = [Reactant("H2", 4.0, 2.016, 2)]
    state.result = "old result"
    state.reaction_energy = -20.0
    page = clear_reaction(state)
    assert_equal(page is not None, True)
    assert_equal(state, make_initial_state())



# ==========================================
# START WEBSITE
# ==========================================

# Run the Bakery tests when executing main.py directly.
# No separate test_main.py is needed; every test is beside its function.
if __name__ == "__main__":
    test_make_initial_state()
    test_count_atoms()
    test_calculate_molar_mass()
    test_get_species()
    test_split_equation()
    test_is_balanced()
    test_find_coefficient()
    test_calculate_chemistry()
    test_index()
    test_reaction_setup()
    test_save_equation()
    test_reactant_input()
    test_save_reactant_mass()
    test_product_input()
    test_finish_calculation()
    test_results()
    test_view_reactants()
    test_thermochemistry()
    test_save_energy()
    test_learn_more()
    test_clear_reaction()

start_server(make_initial_state())
