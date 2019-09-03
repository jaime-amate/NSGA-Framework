import copy
import numpy

def generate_reference_points(nobj, p, scaling=None):
    """Generate reference points uniformly on the hyperplane intersecting
    each axis at 1. The scaling factor is used to combine multiple layers of
    reference points.
    """
    def gen_refs_recursive(ref, nobj, left, total, depth):
        points = []
        if depth == nobj - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                points.extend(gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1))
        return points

    ref_points = numpy.array(gen_refs_recursive(numpy.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / nobj

    return ref_points