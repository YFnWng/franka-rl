"""Package rebuilt Franky with byte-identical native dependencies from a tested venv.

No downloaded/rebuilt dependency enters the output. The built Franky core and
Python extension are patched to resolve exclusively to the copied wheel libraries.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-site', type=Path, required=True)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--template-wheel', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--patchelf', default='patchelf')
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    version=a.version
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(a.template_wheel) as z: z.extractall(root)
        old_info=next(root.glob('*.dist-info')); new_info=root/f'franky_control-{version}.dist-info'
        old_version=old_info.name[len('franky_control-'):-len('.dist-info')]
        old_info.rename(new_info)
        metadata=new_info/'METADATA'
        metadata.write_text(metadata.read_text().replace('Version: '+old_version+'\n','Version: '+version+'\n',1))
        # Existing generated stubs describe an older interface; do not ship stale ones.
        for f in (root/'franky').glob('*.pyi'): f.unlink()
        shutil.rmtree(root/'franky_control.libs')
        shutil.copytree(a.baseline_site/'franky_control.libs',root/'franky_control.libs')
        for name in ('libfranky.so', '_franky.cpython-312-x86_64-linux-gnu.so'):
            target=root/'franky'/name; shutil.copy2(a.build_dir/name,target)
            needed=subprocess.check_output([a.patchelf,'--print-needed',str(target)],text=True).splitlines()
            for dep in needed:
                matches=[]
                if dep.startswith('libfranka'): matches=list((root/'franky_control.libs').glob('libfranka-*.so*'))
                elif dep.startswith('libruckig'): matches=list((root/'franky_control.libs').glob('libruckig-*.so*'))
                if matches:
                    if len(matches)!=1: raise ValueError('Ambiguous dependency '+dep)
                    subprocess.run([a.patchelf,'--replace-needed',dep,matches[0].name,str(target)],check=True)
            subprocess.run([a.patchelf,'--set-rpath','$ORIGIN:$ORIGIN/../franky_control.libs',str(target)],check=True)
        receipt={f.name:digest(f) for f in sorted((root/'franky_control.libs').iterdir()) if f.is_file()}
        for name,sha in receipt.items():
            assert digest(a.baseline_site/'franky_control.libs'/name)==sha
        records=[]
        for f in sorted(root.rglob('*')):
            if f.is_file() and f.name!='RECORD':
                data=f.read_bytes(); h=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
                records.append([str(f.relative_to(root)),'sha256='+h,str(len(data))])
        records.append([str((new_info/'RECORD').relative_to(root)),'',''])
        buf=io.StringIO();csv.writer(buf).writerows(records);(new_info/'RECORD').write_text(buf.getvalue())
        wheel=a.output_dir/f'franky_control-{version}-cp312-cp312-manylinux_2_35_x86_64.whl'
        if wheel.exists(): raise FileExistsError(wheel)
        with zipfile.ZipFile(wheel,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(root.rglob('*')):
                if f.is_file(): z.write(f,f.relative_to(root))
    manifest={'baseline_site':str(a.baseline_site),'wheel':wheel.name,'wheel_sha256':digest(wheel),
              'dependencies':receipt,'dependency_policy':'all bundled native dependencies are byte-identical to baseline'}
    (a.output_dir/'dependency-receipt.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(wheel)


if __name__=='__main__': main()
