"""Pseudonymisation: replace patient identifiers with placeholders before text leaves this machine.

    "Patient Name: Smith, John MRN: 1234567"   ->   "Patient Name: [PERSON_1] MRN: [MRN_1]"

The LLM (and Langfuse) only ever see the placeholders. The mapping stays in this process's memory,
and unmask() puts the real values back in the answer that the clinician reads.

Honest limits (see the notes in the lessons):
- It finds identifiers by LABEL ("Patient Name:", "MRN:", "DOB:") and by shape (phone, email).
  A name with no label in front of it, or a label scrambled by OCR, is NOT caught.
- Only patient identifiers are masked. Clinician names and clinical dates are left alone on purpose,
  because the clinician needs them.
- This reduces exposure. It does not make the system safe for real patient data.
"""
import re
import threading


class Pseudonymizer:
    def __init__(self):
        self._placeholder_of = {}   # real value -> "[PERSON_1]"
        self._real_of = {}          # "[PERSON_1]" -> real value (first form seen)
        self._counts = {}
        self._person_names = set()  # every form of every person name seen so far
        self._lock = threading.Lock()

    def _placeholder(self, kind: str, value: str, alias_of: str | None = None) -> str:
        """The same value always gets the same placeholder. alias_of lets 'John Smith' share
        the placeholder of 'Smith, John'."""
        with self._lock:
            if alias_of and alias_of in self._placeholder_of:
                self._placeholder_of[value] = self._placeholder_of[alias_of]
            if value not in self._placeholder_of:
                self._counts[kind] = self._counts.get(kind, 0) + 1
                ph = f"[{kind}_{self._counts[kind]}]"
                self._placeholder_of[value] = ph
                self._real_of[ph] = value
            if kind == "PERSON":
                self._person_names.add(value)
            return self._placeholder_of[value]

    # ---- masking ----
    NAME_LABEL = re.compile(r"\b(?:Patient Name|Patient|Name)\s*:\s*([A-Z][\w'’-]+(?:,\s*|\s+)[A-Z][\w'’-]+)")
    MRN = re.compile(r"(\bMRN\s*:?\s*)(\d{4,})")
    VISIT = re.compile(r"(\bVisit Number\s*:?\s*)(\d{6,})")
    DOB = re.compile(
        r"(\bDOB\s*:?\s*)(\d{1,2}[-/ ][A-Za-z]{3}[-/ ]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})"
    )
    PHONE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
    EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

    def mask(self, text: str) -> str:
        # 1. Names. A label ("Patient Name:", "Patient:", "Name:") teaches us a name; after that we mask
        #    EVERY occurrence of it, in both orders ("Smith, John" and "John Smith"), in this text and
        #    in any text we see later.
        for name in dict.fromkeys(self.NAME_LABEL.findall(text)):
            part1, part2 = (p.strip() for p in re.split(r",\s*|\s+", name, maxsplit=1))
            swapped = f"{part2} {part1}" if "," in name else f"{part2}, {part1}"
            self._placeholder("PERSON", name)
            self._placeholder("PERSON", swapped, alias_of=name)
        with self._lock:
            known = sorted(self._person_names, key=len, reverse=True)
            lookup = {v: self._placeholder_of[v] for v in known}
        for variant in known:
            text = re.sub(r"(?<![\w])" + re.escape(variant) + r"(?![\w])", lookup[variant], text)

        # 2. Identifiers that have a label in front of them.
        text = self.MRN.sub(lambda m: m.group(1) + self._placeholder("MRN", m.group(2)), text)
        text = self.VISIT.sub(lambda m: m.group(1) + self._placeholder("VISIT", m.group(2)), text)
        text = self.DOB.sub(lambda m: m.group(1) + self._placeholder("DOB", m.group(2)), text)

        # 3. Identifiers recognised by their shape.
        text = self.PHONE.sub(lambda m: self._placeholder("PHONE", m.group(0)), text)
        text = self.EMAIL.sub(lambda m: self._placeholder("EMAIL", m.group(0)), text)
        return text

    # ---- restoring ----
    def unmask(self, text: str) -> str:
        return re.sub(r"\[[A-Z]+_\d+\]", lambda m: self._real_of.get(m.group(0), m.group(0)), text)


_default = Pseudonymizer()
mask = _default.mask
unmask = _default.unmask