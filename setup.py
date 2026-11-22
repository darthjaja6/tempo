import pathlib
import shutil
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class CUDABuildExt(_build_ext):
    def build_extension(self, ext: Extension) -> None:
        root = pathlib.Path(__file__).resolve().parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import build_ext as builder

        built_so = builder.build()
        dst = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if built_so.resolve() != dst:
            shutil.copy2(built_so, dst)


setup(
    ext_modules=[Extension("tempo._tempo_C", sources=[])],
    cmdclass={"build_ext": CUDABuildExt},
)
