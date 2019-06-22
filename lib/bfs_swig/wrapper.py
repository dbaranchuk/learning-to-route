"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
import os
import os.path as osp
import setuptools.sandbox

package_abspath = osp.join(*osp.split(osp.abspath(__file__))[:-1])
if not os.path.exists(osp.join(package_abspath, '_bfs.so')):
    # try build _bfs.so
    workdir = os.getcwd()
    try:
        os.chdir(package_abspath)
        setuptools.sandbox.run_setup(osp.join(package_abspath, 'setup.py'), ['clean', 'build'])
        os.system('cp {}/build/lib*/*.so {}/_bfs.so'.format(package_abspath, package_abspath))
        assert os.path.exists(osp.join(package_abspath, '_bfs.so'))
    finally:
        os.chdir(workdir)

try:
    from . import bfs as bfs_module
except ImportError:
    from . import _bfs as bfs_module

bfs = bfs_module.bfs_visited_ids
