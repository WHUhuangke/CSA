from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read().splitlines()

setup(
    name="CSA",  # 替换为你的包名
    version="0.1.0",           # 初始版本号
    author="Ke Huang",
    author_email="2021302111027@whu.edu.cn",
    description="Spatial transcriptomic data "
                "denoising and domain identification "
                "by a community strength-augmented graph autoencoder",
    long_description=long_description,
    long_description_content_type="markdown",
    url="https://github.com/WHUhuangke/CSA",  # 项目主页
    packages=find_packages(),  # 自动发现所有包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",     # 支持的Python版本
    install_requires=install_requires,  # 从requirements.txt获取依赖
    entry_points={
        "console_scripts": [
            "your-command=your_package.module:main_function",  # 命令行工具
        ],
    },
)