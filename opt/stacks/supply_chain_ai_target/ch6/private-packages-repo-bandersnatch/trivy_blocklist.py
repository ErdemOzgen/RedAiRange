import json
import subprocess
from bandersnatch.filter import FilterProjectPlugin

class MyBlocklist(FilterProjectPlugin):
    name = "my_blocklist"
    blacklisted_packages = []

    async def filter(self, info):
        package = info["name"]
        version = info["info"]["version"]
        package_file = f"{package}-{version}.tar.gz"  # Assuming tar.gz files, adapt as needed

        # Run Trivy on the package
        result = subprocess.run(
            ["trivy", "fs", f"/data/pypi/packages/{package_file}"],
            capture_output=True,
            text=True
        )

        vulnerabilities = json.loads(result.stdout)

        for vuln in vulnerabilities:
            severity = vuln.get("Severity")
            if severity in ["HIGH", "CRITICAL"]:
                self.logger.info(f"Blocking package {package} version {version} due to {severity} vulnerability")
                self.blacklisted_packages.append(package)
                return True

        return False

# At the end of your Bandersnatch run, you can generate a timestamped report for blacklisted packages.
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
with open(f"blacklist_report_{timestamp}.txt", "w") as f:
    f.write("\n".join(MyBlocklist.blacklisted_packages))
