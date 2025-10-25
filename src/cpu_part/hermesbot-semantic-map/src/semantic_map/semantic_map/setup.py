from setuptools import setup, find_packages

package_name = "semantic_map"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(include=["semantic_map", "semantic_map.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Artem Voronov",
    maintainer_email="artem_voronov@skoltech.com",
    description="Semantic Map that combines the SLAM and object detection nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "semantic_map_3d = semantic_map.semantic_map_3d:main"
        ],
    },
)
