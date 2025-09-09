from typing import Dict, Any, Tuple
from .exceptions_file import DataTransformationError

def check_schema(row: Dict[str, Any], schema: Dict[str, type], nullable:set = set()) -> Tuple[bool, str]:
    try:
        for col, col_type in schema.items():
            if col not in row:
                return False, f"Missing column: {col}"
            if row[col] is None and col in nullable:
                continue
            if not isinstance(row[col], col_type) and row[col] is not None:
                return False, f"Invalid type for column: {col}: expected {col_type}, got {type(row[col])}"
        return True, ""
    except Exception as e:
        raise DataTransformationError(f"Schema validation error: {e}")




def check_fk(row: Dict[str, Any], fk_col: str, valid_ids: set) -> Tuple[bool, str]:
    try:
        if fk_col not in row:
            return False, f"Missing FK column: {fk_col}"
        if row[fk_col] not in valid_ids:
            return False, f"Invalid FK value in column: {fk_col}"
        return True, ""
    except Exception as e:
        raise DataTransformationError(f"FK validation error: {e}")
    

    