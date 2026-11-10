from agent.privacy import Pseudonymizer


def test_identifiers_are_masked():
    out = Pseudonymizer().mask("Patient Name: Smith, John MRN: 1234567 Phone 416-555-5555")
    assert "Smith" not in out and "1234567" not in out and "416-555-5555" not in out


def test_unmask_restores_the_original():
    p = Pseudonymizer()
    original = "Patient Name: Smith, John MRN: 1234567"
    assert p.unmask(p.mask(original)) == original


def test_same_person_gets_the_same_placeholder_in_both_orders():
    p = Pseudonymizer()
    p.mask("Patient Name: Smith, John")
    assert p.mask("John Smith was seen") == p.mask("Smith, John was seen")


def test_guideline_text_is_unchanged():
    text = "Give 70 ml/kg of Ringer's lactate over 5 h."
    assert Pseudonymizer().mask(text) == text