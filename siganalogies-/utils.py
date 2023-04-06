import random
"""File containing generic data manipulation methods for analogy-oriented datasets."""

def enrich(a, b, c, d):
    """Apply the example generation process from 'Solving Word Analogies: a Machine Learning Perspective'.
    
    For a given positive (i.e. valid) analogy A:B::C:D, this function yields the folowing permutations corresponding to
    positive analogies:
        - A:B::C:D (base form);
        - C:D::A:B (symmetry);
        - B:A::D:C (inside pair reversing);
        - D:C::B:A (symmetry + inside pair reversing);
        - A:C::B:D (central permutation);
        - C:A::D:B (central permutation + inside pair reversing = extreme permutation);
        - B:D::A:C (central permutation + symmetry);
        - D:B::C:A (extreme permutation + symmetry).
    """
    yield a, b, c, d
    yield c, d, a, b
    yield b, a, d, c
    yield d, c, b, a
    yield a, c, b, d
    yield c, a, d, b
    yield b, d, a, c
    yield d, b, c, a

def generate_negative(a, b, c, d):
    """Apply the negative example generation process from 'Solving Word Analogies: a Machine Learning Perspective'.

    For a given positive (i.e. valid) analogy A:B::C:D, this function yields the folowing permutations corresponding to
    negative (i.e. invalid) analogies:
        - B:A::C:D (one pair internally permuted, but not the other);
        - C:B::A:D (analogy between the "diagonals" of the "parallelogram" ABCD);
        - A:A::C:D (one pair is the identity, the other is not).
    """
    yield b, a, c, d
    yield c, b, a, d
    yield a, a, c, d

def random_sample_negative(a, b, c, d, filter_invalid=True, tensors=True, n=8):
    """Randomly sample invalid analogies from the available negative forms.
    
    In this variant, central permutation is considered a property of analogy.
    
    :param filter_invalid: If True, will exclude quadruples from negative forms if the an identical quadruple is already present in the positive forms.
    :param tensors: If True, consider that a, b, c, and d are tensors and use an appropriate equality check for quadruples.
    :param n: Number of valid and invalid forms to return. If None, all the forms will be returned, no matter the number.
    """
    # random sampling
    return n_pos_n_neg(a, b, c, d, filter_invalid=filter_invalid, tensors=tensors, n=n)[1]

def n_pos_n_neg(a, b, c, d, filter_invalid=True, filter_valid=False, tensors=True, n=8):
    """Generate positive examples (including the base form) and negative examples form a base analogical form.
    
    In this variant, central permutation is considered a property of analogy.

    Returns two lists, one of positive forms, the other of negative forms.

    :param filter_invalid: If True, will exclude quadruples from negative forms if the an identical quadruple is already present in the positive forms.
    :param filter_valid: If True, will exclude quadruples from positive forms if the an identical quadruple is already present in the negative forms.
        Cannot be True if filter_invalid is True
    :param tensors: If True, consider that a, b, c, and d are tensors and use an appropriate equality check for quadruples.
    :param n: Number of valid and invalid forms to return. If None, all the forms will be returned, no matter the number.
    """

    assert not (filter_invalid and filter_valid), "filter_invalid and filter_valid must not be True together."

    def tensor_in(a, b, c, d, list_of_form):
        # check if the quadruple is equal to a quadruple in the givenlist of forms
        return any(
            a.equal(a_) and
            b.equal(b_) and
            c.equal(c_) and
            d.equal(d_)
            for a_, b_, c_, d_ in list_of_form
        )

    # Case 1: filtering out invalid forms which are already present as valid forms
    if filter_invalid:
        positive_forms = [(a_, b_, c_, d_) for a_, b_, c_, d_ in enrich(a, b, c, d)]
        
        negative_forms=[(a__, b__, c__, d__)
            for a_, b_, c_, d_ in positive_forms # for each positive (valid) analogies
            for a__, b__, c__, d__ in generate_negative(a_, b_, c_, d_) # generate 3 negative (invalid) analogies
            if (
                # remove potentially problematic forms (which can appear when the input is a:b:a:b for example)
                (tensors and not tensor_in(a__, b__, c__, d__, positive_forms)) or 
                (not tensors and (a__, b__, c__, d__) not in positive_forms)
            )
        ]

    # Case 2: filtering out valid forms which are already present as invalid forms
    elif filter_valid:
        positive_forms = [(a_, b_, c_, d_) for a_, b_, c_, d_ in enrich(a, b, c, d)]
        
        negative_forms = [(a__, b__, c__, d__)
            for a_, b_, c_, d_ in positive_forms # for each positive (valid) analogies
            for a__, b__, c__, d__ in generate_negative(a_, b_, c_, d_) # generate 3 negative (invalid) analogies
        ]
        positive_forms = [
            (a_, b_, c_, d_) for a_, b_, c_, d_ in positive_forms
            if (
                # remove potentially problematic forms (which can appear when the input is a:b:a:b for example)
                (tensors and not tensor_in(a_, b_, c_, d_, negative_forms)) or 
                (not tensors and (a_, b_, c_, d_) not in negative_forms)
            )
        ]

    # Case 3: no filtering
    else:
        positive_forms = [(a_, b_, c_, d_) for a_, b_, c_, d_ in enrich(a, b, c, d)]
        negative_forms = [(a__, b__, c__, d__)
            for a_, b_, c_, d_ in positive_forms # for each positive (valid) analogies
            for a__, b__, c__, d__ in generate_negative(a_, b_, c_, d_) # generate 3 negative (invalid) analogies
        ]

    # Sampling the right number
    if n is None or len(positive_forms) == n:
        positive_forms = positive_forms
    elif len(positive_forms) > n:
        positive_forms = random.sample(positive_forms, n)
    else: # len(positive_forms) < n:
        positive_forms = positive_forms + random.choices(positive_forms, k=n - len(positive_forms))

    # Sampling the right number
    if n is None or len(negative_forms) == n:
        negative_forms = negative_forms
    elif len(negative_forms) > n:
        negative_forms = random.sample(negative_forms, n)
    else: # len(negative_forms) < n:
        negative_forms = negative_forms + random.choices(negative_forms, k=n - len(negative_forms))

    return positive_forms, negative_forms