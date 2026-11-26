from typing import Any

from sqlalchemy import ForeignKey, LargeBinary, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    # schema.sql owns DDL; this base maps existing tables for query/insert use.
    pass


class ModuleInstance(Base):
    # A module row represents one initialized state, not just a reusable architecture.
    __tablename__ = 'module_instance'

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    target: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    module_config: Mapped[list['ModuleConfig']] = relationship('ModuleConfig', back_populates='module')
    test_case: Mapped[list['TestCase']] = relationship('TestCase', back_populates='module')


class ModuleConfig(Base):
    # schema.sql creates this as STRICT ANY. Text supplies SQLAlchemy's scalar binding;
    # schema.sql remains the DDL source so CREATE TABLE retains SQLite's native ANY type.
    __tablename__ = 'module_config'

    module_id: Mapped[str] = mapped_column(ForeignKey('module_instance.id'), primary_key=True)
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[Any] = mapped_column(Text)

    module: Mapped['ModuleInstance'] = relationship('ModuleInstance', back_populates='module_config')


class TestCase(Base):
    # Cases reference a concrete initialized module and own named forward arguments.
    __tablename__ = 'test_case'

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    module_id: Mapped[str] = mapped_column(ForeignKey('module_instance.id'), nullable=False)

    module: Mapped['ModuleInstance'] = relationship('ModuleInstance', back_populates='test_case')
    test_case_input: Mapped[list['TestCaseInput']] = relationship('TestCaseInput', back_populates='case')


class TestCaseExpected(Base):
    # Expected output is one binary tensor bundle for each case, not a forward argument.
    __tablename__ = 'test_case_expected'

    case_id: Mapped[str] = mapped_column(ForeignKey('test_case.id'), primary_key=True)
    value: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class TestCaseInput(Base):
    # The BLOB contains a Safetensors payload; the row name is the forward() argument.
    __tablename__ = 'test_case_input'

    case_id: Mapped[str] = mapped_column(ForeignKey('test_case.id'), primary_key=True)
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    case: Mapped['TestCase'] = relationship('TestCase', back_populates='test_case_input')
