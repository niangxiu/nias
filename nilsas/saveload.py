import os
import pickle
from collections import namedtuple

Segment = namedtuple('Segment', 'u f fs J adjointbundle') # avb is acronym for adjoint vector bundle

def save_segment(sg_path, sg):
    '''
    save a segment file under the path sg_path,
    naming convention is mXX_segmentYYY, where XX and YY are given by cp.lss
    '''
    filename = 'm{0}_segment{1}'.format(sg.adjointbundle.m_modes(), cp.adjointbundle.K_segments())
    with open(os.path.join(checkpoint_path, filename), 'wb') as f:
        pickle.dump(sg, f)

def load_segment(sg_file):
    return pickle.load(open(sg_file, 'rb'))

def verify_checkpoint(segment):
    u, f, fs, J, adjointbundle = segment
    return adjointbundle.K_segments() == len(u) \
                            == len(f) \
                            == len(fs) \
                            == len(J)

def load_last_segment(segment_path, m):
    '''
    load segment in path segment_path, with file name mXX_segmentYYY,
    where XX matches the given m, and YY is the largest
    '''
    def m_modes(filename):
        try:
            m, _ = filename.split('_segment')
            assert m.startswith('m')
            return int(m[1:])
        except:
            return None

    def segments(filename):
        try:
            _, segments = filename.split('_segment')
            return int(segments)
        except:
            return None

    files = filter(lambda f : m_modes(f) == m and segments(f) is not None,
                   os.listdir(segment_path))
    files = sorted(files, key=segments)
    if len(files):
        return load_checkpoint(os.path.join(segment_path, files[-1]))
