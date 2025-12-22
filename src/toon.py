"""
TOON Format Implementation v3.0
Token-Oriented Object Notation - Compliant with official specification.

Based on: https://toonifyit.com/docs
"""

import re
import io
from typing import List, Dict, Any, Optional, Union, Tuple

class ToonError(Exception):
    """TOON parsing or encoding error."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# ENCODER
# ═══════════════════════════════════════════════════════════════════════════

def encode(
    value: Any,
    indent: int = 2,
    delimiter: str = ",",
    length_marker: bool = False,
    key_folding: str = "off",  # "off" | "safe"
    flatten_depth: int = 999,  # Effectively infinite by default
    _level: int = 0,
) -> str:
    """
    Encode any JSON-serializable value to TOON format.
    
    Args:
        value: JSON-serializable value (dict, list, primitives)
        indent: Spaces per indentation level (default: 2)
        delimiter: Array delimiter: ',' (default), '\t', '|'
        length_marker: Add # prefix to array lengths
        key_folding: "safe" to fold single-key nested objects into dotted keys
        flatten_depth: Max depth for key folding
        _level: Internal - current indentation level
    
    Returns:
        TOON-formatted string
    """
    if value is None:
        return "null"
    
    if isinstance(value, bool):
        return "true" if value else "false"
    
    if isinstance(value, (int, float)):
        return str(value)
    
    if isinstance(value, str):
        return _escape_string(value, delimiter)
    
    if isinstance(value, list):
        return _encode_array(value, indent, delimiter, length_marker, key_folding, flatten_depth, _level)
    
    if isinstance(value, dict):
        return _encode_object(value, indent, delimiter, length_marker, key_folding, flatten_depth, _level)
    
    # Fallback for other types
    return _escape_string(str(value), delimiter)


def _encode_object(
    obj: dict,
    indent: int,
    delimiter: str,
    length_marker: bool,
    key_folding: str,
    flatten_depth: int,
    level: int,
) -> str:
    """Encode a dictionary to TOON."""
    if not obj:
        return ""
    
    # Apply key folding if enabled
    if key_folding == "safe":
        obj = _fold_object_keys(obj, flatten_depth, delimiter)
    
    lines = []
    ind = " " * (indent * level)
    
    for key, value in obj.items():
        # Key encoding
        key_str = str(key)
        if _needs_quoting(key_str, delimiter, is_key=True):
            key_str = _escape_string(key_str, delimiter)
        
        prefix = f"{ind}{key_str}:"
        
        if value is None:
            lines.append(f"{prefix} null")
        elif isinstance(value, bool):
            lines.append(f"{prefix} {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{prefix} {value}")
        elif isinstance(value, str):
            escaped = _escape_string(value, delimiter)
            lines.append(f"{prefix} {escaped}")
        elif isinstance(value, list):
            arr_str = _encode_array_value(key_str, value, indent, delimiter, length_marker, key_folding, flatten_depth, level)
            lines.append(arr_str)
        elif isinstance(value, dict):
            if not value:
                lines.append(f"{prefix}")
            else:
                nested = _encode_object(value, indent, delimiter, length_marker, key_folding, flatten_depth, level + 1)
                lines.append(f"{prefix}")
                lines.append(nested)
        else:
            lines.append(f"{prefix} {_escape_string(str(value), delimiter)}")
    
    return "\n".join(lines)


def _fold_object_keys(obj: dict, max_depth: int, delimiter: str) -> dict:
    """Fold nested single-key objects into dotted keys."""
    new_obj = {}
    for key, value in obj.items():
        current_key = str(key)
        current_val = value
        depth = 1
        
        # Determine strict eligibility for folding
        # Must be safe identifier, no quoting needed, single key in nested obj
        while (
            isinstance(current_val, dict) 
            and len(current_val) == 1 
            and depth < max_depth
        ):
            next_key = list(current_val.keys())[0]
            next_val = list(current_val.values())[0]
            
            # Check eligibility of keys
            if (
                _is_identifier(current_key) and 
                _is_identifier(str(next_key)) and 
                not _needs_quoting(current_key, delimiter, is_key=True) and
                not _needs_quoting(str(next_key), delimiter, is_key=True)
            ):
                current_key = f"{current_key}.{next_key}"
                current_val = next_val
                depth += 1
            else:
                break
        
        new_obj[current_key] = current_val
    return new_obj


def _is_identifier(s: str) -> bool:
    """Check if string is a safe identifier."""
    # Strict 1.9 definition: letters, digits, underscores, no start digit, no dots
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s))


def _encode_array_value(
    key: str,
    arr: list,
    indent: int,
    delimiter: str,
    length_marker: bool,
    key_folding: str,
    flatten_depth: int,
    level: int,
) -> str:
    """Encode an array as a value of an object key."""
    ind = " " * (indent * level)
    child_ind = " " * (indent * (level + 1))
    length_prefix = "#" if length_marker else ""
    
    # 1. Empty
    if not arr:
        return f"{ind}{key}[{length_prefix}0]:"
    
    # 2. Tabular (v3.0 §9.3)
    if _is_tabular_array(arr):
        return _encode_tabular_array(key, arr, indent, delimiter, length_marker, level)
    
    # 3. Inline Primitive (v3.0 §9.1)
    if _is_primitive_array(arr):
        values = [_encode_primitive(v, delimiter) for v in arr]
        return f"{ind}{key}[{length_prefix}{len(arr)}]: {delimiter.join(values)}"
    
    # 4. Mixed/Expanded List (v3.0 §9.4)
    lines = [f"{ind}{key}[{length_prefix}{len(arr)}]:"]
    for item in arr:
        if isinstance(item, dict):
            # Object as List Item (§10)
            if not item:
                lines.append(f"{child_ind}-")
            else:
                items_list = list(item.items())
                first_k, first_v = items_list[0]
                
                # Special Case: First field is Multi-line Tabular Array?
                # Per §10: If first field is tabular array, header goes on hyphen line
                if isinstance(first_v, list) and _is_tabular_array(first_v):
                    # Render tabular header on "- field[N]{cols}:" line
                    tab_header_obj = _encode_tabular_array(str(first_k), first_v, indent, delimiter, length_marker, level + 1)
                    # Strip the indent from the header since we place it after "- "
                    tab_lines = tab_header_obj.split("\n")
                    header_line = tab_lines[0].strip()
                    row_lines = tab_lines[1:]
                    
                    lines.append(f"{child_ind}- {header_line}")
                    # Rows are at +2 depth relative to hyphen line (level + 1 + 2? No, relative to hyphen)
                    # Code says "Tabular rows MUST appear at depth +2 (relative to the hyphen line)."
                    # list item hyphen is at `child_ind` (level+1).
                    # rows should be at level+3.
                    # Current `_encode_tabular_array` called with level+1 produces indent level+1 for header, level+2 for rows.
                    # We need rows at level+3. So let's re-indent rows.
                    for r in row_lines:
                        # r is indented at level+2. We need level+3.
                        lines.append(f"{' ' * indent}{r}")

                # Standard Case: First field on hyphen line
                else:
                    first_k_str = str(first_k)
                    if _needs_quoting(first_k_str, delimiter, is_key=True):
                        first_k_str = _escape_string(first_k_str, delimiter)
                        
                    prefix = f"{child_ind}- {first_k_str}:"
                    
                    if isinstance(first_v, (dict, list)):
                        # Complex first value -> open on new line
                        lines.append(prefix)
                        nested = encode(first_v, indent, delimiter, length_marker, key_folding, flatten_depth, level + 2)
                        lines.append(nested)
                    else:
                        # Primitive first value -> inline
                        encoded_val = _encode_primitive(first_v, delimiter)
                        lines.append(f"{prefix} {encoded_val}")
                
                # Remaining encoded fields
                for k, v in items_list[1:]:
                    # Standard object encoding for siblings
                    sib_lines = _encode_object({k: v}, indent, delimiter, length_marker, key_folding, flatten_depth, level + 2)
                    lines.append(sib_lines)
                    
        else:
            # Primitive/Array as list item
            if isinstance(item, list):
                 # Nested array
                 # "- [M]: ..."
                 nested = _encode_array(item, indent, delimiter, length_marker, key_folding, flatten_depth, 0) # 0 level, handle indent manually
                 lines.append(f"{child_ind}- {nested}")
            else:
                lines.append(f"{child_ind}- {_encode_primitive(item, delimiter)}")
    
    return "\n".join(lines)


def _encode_array(
    arr: list,
    indent: int,
    delimiter: str,
    length_marker: bool,
    key_folding: str,
    flatten_depth: int,
    level: int,
) -> str:
    """Encode a standalone array (not as object value)."""
    length_prefix = "#" if length_marker else ""
    ind = " " * (indent * level)
    
    if not arr:
        return f"[{length_prefix}0]:"
    
    if _is_primitive_array(arr):
        values = [_encode_primitive(v, delimiter) for v in arr]
        return f"[{length_prefix}{len(arr)}]: {delimiter.join(values)}"
    
    if _is_tabular_array(arr):
        keys = list(arr[0].keys())
        header_delim = ""
        if delimiter == "\t": header_delim = "\\t"
        elif delimiter == "|": header_delim = "|"
        
        # Note: Tabular root array omits "key"
        sep = delimiter if delimiter != "," else ","
        f_header = sep.join(keys)
        # Root tabular format: "[N]{fields}:"
        lines = [f"[{length_prefix}{len(arr)}{header_delim}]{{{f_header}}}:"]
        
        for item in arr:
            values = [_encode_primitive(item.get(k), delimiter) for k in keys]
            lines.append(f"{' ' * indent}{delimiter.join(values)}")
        return "\n".join(lines)
    
    # Mixed array
    lines = [f"[{length_prefix}{len(arr)}]:"]
    for item in arr:
        encoded = encode(item, indent, delimiter, length_marker, key_folding, flatten_depth, level + 1)
        lines.append(f"{' ' * indent}- {encoded}")
    return "\n".join(lines)


def _encode_tabular_array(
    key: str,
    arr: list,
    indent: int,
    delimiter: str,
    length_marker: bool,
    level: int,
) -> str:
    """Encode a uniform array of objects in tabular format."""
    ind = " " * (indent * level)
    child_ind = " " * (indent * (level + 1))
    length_prefix = "#" if length_marker else ""
    
    keys = list(arr[0].keys())
    
    if delimiter == "\t": header_delim = "\\t"
    elif delimiter == "|": header_delim = "|"
    else: header_delim = ""
    
    sep = delimiter if delimiter != "," else ","
    f_header = sep.join(keys)
    
    header = f"{ind}{key}[{length_prefix}{len(arr)}{header_delim}]{{{f_header}}}:"
    lines = [header]
    
    for item in arr:
        values = [_encode_primitive(item.get(k), delimiter) for k in keys]
        lines.append(f"{child_ind}{delimiter.join(values)}")
    
    return "\n".join(lines)


def _is_tabular_array(arr: list) -> bool:
    """Check if array can use tabular format."""
    if not arr or not isinstance(arr[0], dict):
        return False
    if not arr[0]: return False
    first_keys = set(arr[0].keys())
    for item in arr:
        if not isinstance(item, dict): return False
        if set(item.keys()) != first_keys: return False
        for v in item.values():
            if isinstance(v, (dict, list)): return False
    return True


def _is_primitive_array(arr: list) -> bool:
    """Check if array contains only primitives."""
    for item in arr:
        if isinstance(item, (dict, list)): return False
    return True


def _encode_primitive(value: Any, delimiter: str) -> str:
    if value is None: return "null"
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, (int, float)): return str(value)
    s_val = str(value)
    return _escape_string(s_val, delimiter)


def _needs_quoting(value: str, delimiter: str, is_key: bool = False) -> bool:
    """Check if string needs quoting per TOON spec v3.0."""
    if not value: return True
    if value != value.strip(): return True
    
    # §1.9 IdentifierSegment pattern for keys? Spec says keys matching ^[A-Za-z_][A-Za-z0-9_.]*$ can be unquoted
    if is_key:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", value):
            return False
        return True

    if value in ("true", "false", "null"): return True
    # Numeric-like
    if re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", value): return True
    # Leading zero integers
    if re.match(r"^0\d+$", value): return True
    
    special = f':"{delimiter}\n\r\t\\[]{{}}' 
    if any(c in value for c in special): return True
    
    if value.startswith("-"): return True
    if re.match(r"^\[\d+\]", value): return True
    
    return False


def _escape_string(value: str, delimiter: str) -> str:
    """Escape and quote a string."""
    # Always check quoting rules first
    if not _needs_quoting(value, delimiter):
        return value
        
    out = io.StringIO()
    out.write('"')
    for char in value:
        if char == '\\': out.write('\\\\')
        elif char == '"': out.write('\\"')
        elif char == '\n': out.write('\\n')
        elif char == '\r': out.write('\\r')
        elif char == '\t': out.write('\\t')
        else: out.write(char)
    out.write('"')
    return out.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════════════════

def decode(
    text: str,
    indent: int = 2,
    strict: bool = True,
    expand_paths: str = "off", # "off" | "safe"
) -> Any:
    """
    Decode a TOON-formatted string to Python values.
    
    Args:
        text: TOON-formatted string
        indent: Expected indentation (default: 2)
        strict: Enable strict validation (default: True)
        expand_paths: "safe" to expand dotted keys into nested objects
    
    Returns:
        Decoded Python value
    """
    text = _clean_input(text)
    if not text:
        return {}
    
    lines = text.split("\n")
    
    # Check for empty doc
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return {}
        
    # Root form discovery (§5)
    first = non_empty[0].strip()
    
    # 1. Root Array Header [N]...
    if re.match(r"^\[#?\d+", first):
         val = _decode_standalone_array(lines, indent, strict)
    # 2. Tabular Root with Key key[N]...
    elif re.match(r"^\w+(?:[\.\w]+)*\[#?\d+", first):
         val = _decode_tabular_root(lines, indent, strict)
    # 3. Single Primitive (one line, no colon, no header)
    elif len(non_empty) == 1 and ":" not in first and not _is_header(first):
         val = _parse_value(first)
    # 4. Object
    else:
         val, _ = _decode_object(lines, 0, indent, strict)
         
    # Apply Path Expansion (§13.4) if enabled
    if expand_paths == "safe":
        val = _expand_paths(val, strict)
        
    return val


def _expand_paths(val: Any, strict: bool) -> Any:
    """Post-processing step to expand dotted keys."""
    if isinstance(val, dict):
        new_obj = {}
        for k, v in val.items():
            # Recursively expand children first
            v_expanded = _expand_paths(v, strict)
            
            # Check for expansion eligibility
            k_str = str(k)
            if "." in k_str:
                parts = k_str.split(".")
                if all(_is_identifier(p) for p in parts):
                    # Expand
                    current = new_obj
                    for i, part in enumerate(parts[:-1]):
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                        if not isinstance(current, dict):
                            if strict:
                                raise ToonError(f"Expansion conflict at path '{part}' (object vs primitive)")
                            # LWW: Overwrite with object is implied if we continue, but we need to handle existing
                            current = {} # Reset to object? Or just fail? LWW says "last write wins".
                            # Here we are merging into 'new_obj'. Later keys overwrite.
                    
                    # Leaf
                    last_part = parts[-1]
                    current[last_part] = v_expanded
                    continue
            
            # No expansion or not eligible
            new_obj[k] = v_expanded
        return new_obj
        
    elif isinstance(val, list):
        return [_expand_paths(x, strict) for x in val]
    
    return val


def _decode_object(lines: List[str], start_level: int, indent: int, strict: bool) -> Tuple[dict, int]:
    """Decode an object."""
    result = {}
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        
        # Skip blank lines (Strict mode check handled externally or here?)
        # §12: Outside arrays, blank lines ignored. Inside, strict error.
        # Here we are decoding an object, so technically "outside" arrays?
        if not line.strip():
            i += 1
            continue
            
        current_indent = len(line) - len(stripped)
        current_level = current_indent // indent if indent > 0 else 0
        
        if current_level < start_level:
            break
        if current_level > start_level:
            # Sibling or child of previous? If we are here, it's NOT a child of previous loop item
            # because previous item would have consumed its children.
            # So this is an indentation error or garbage.
            i += 1
            continue

        # Parse key: value
        if ":" in stripped:
            # Handle headers first
            if _is_header(stripped):
                 # Array field
                 arr_key, count, delim, fields, inline = _parse_header(stripped)
                 if count == 0:
                     result[arr_key] = []
                     i += 1
                 elif inline and not fields:
                     result[arr_key] = _parse_inline_values(inline, delim)
                     i += 1
                 else:
                     # Multiline array (Tabular or Mixed)
                     # Determine expected rows/lines
                     # This logic needs to consume lines until indentation drops back
                     sub_lines = []
                     start_idx = i + 1
                     while start_idx < len(lines):
                         next_l = lines[start_idx]
                         if not next_l.strip(): 
                             start_idx += 1
                             continue
                         next_ind = len(next_l) - len(next_l.lstrip())
                         if next_ind <= current_indent: break
                         sub_lines.append(next_l)
                         start_idx += 1
                     
                     if fields:
                         result[arr_key] = _parse_tabular_rows(sub_lines, fields, delim, count, strict)
                     else:
                         result[arr_key] = _parse_list_array(sub_lines, indent, strict)
                     i = start_idx
            
            else:
                # Regular key: value
                colon_idx = stripped.index(":")
                key = stripped[:colon_idx].strip()
                val_part = stripped[colon_idx+1:].strip()
                
                if not val_part:
                    # Nested object
                    # Consume children
                    sub_lines_start = i + 1
                    sub_lines = []
                    while sub_lines_start < len(lines):
                        next_l = lines[sub_lines_start]
                        if not next_l.strip():
                            sub_lines_start += 1
                            continue
                        if len(next_l) - len(next_l.lstrip()) <= current_indent:
                            break
                        sub_lines.append(next_l)
                        sub_lines_start += 1
                    
                    nested_obj, _ = _decode_object(sub_lines, start_level + 1, indent, strict)
                    result[key] = nested_obj
                    i = sub_lines_start
                else:
                    result[key] = _parse_value(val_part)
                    i += 1
        else:
            # Malformed line "key" without colon? Or just primitive on wrong line?
            if strict:
                raise ToonError(f"Missing colon after key: {stripped}")
            i += 1
            
    return result, i


def _parse_header(line: str) -> Tuple[str, int, str, List[str], str]:
    """Parse header line."""
    # Regex for key[N]{fields}: ... (Allowing spaces for LLM robustness)
    # Groups: 1=key, 2=count, 3=delim, 4=fields, 5=inline
    pattern = r"^((?:[\w\.]+)?)?\s*\[\s*#?\s*(\d+)\s*(\\t|\|)?\s*\]\s*(?:\{\s*([^}]*)\s*\})?\s*:?\s*(.*)$"
    m = re.match(pattern, line.strip())
    if not m:
        raise ToonError(f"Invalid header: {line}")
        
    key = m.group(1) or ""
    count = int(m.group(2))
    delim_char = m.group(3)
    fields_str = m.group(4)
    inline = m.group(5)
    
    delim = ","
    if delim_char == "\\t": delim = "\t"
    elif delim_char == "|": delim = "|"
    elif fields_str and "\t" in fields_str: delim = "\t" # Inference
    
    fields = []
    if fields_str:
        fields = [f.strip() for f in fields_str.split(delim if delim != "," else ",")]
        
    return key, count, delim, fields, inline

def _is_header(line: str) -> bool:
    # Relaxed check
    return bool(re.match(r"^((?:[\w\.]+)?)?\s*\[\s*#?\s*(\d+)", line.strip()))


def _decode_standalone_array(lines: List[str], indent: int, strict: bool) -> list:
    _, count, delim, fields, inline = _parse_header(lines[0])
    if count == 0: return []
    if inline and not fields: return _parse_inline_values(inline, delim)
    
    # Multiline
    if fields:
        return _parse_tabular_rows(lines[1:], fields, delim, count, strict)
    return _parse_list_array(lines[1:], indent, strict)


def _decode_tabular_root(lines: List[str], indent: int, strict: bool) -> list:
    key, count, delim, fields, inline = _parse_header(lines[0])
    if count == 0: return []
    if fields:
        return _parse_tabular_rows(lines[1:], fields, delim, count, strict)
    return []


def _parse_tabular_rows(lines: List[str], fields: List[str], delimiter: str, expected: int, strict: bool) -> list:
    res = []
    for line in lines:
        if not line.strip(): 
            if strict and res and len(res) < expected:
                 raise ToonError(f"Blank line inside partial tabular data")
            continue
            
        vals = _split_row(line.strip(), delimiter)
        if len(vals) != len(fields):
            if strict: raise ToonError(f"Row width mismatch. Expected {len(fields)}, got {len(vals)}")
            while len(vals) < len(fields): vals.append(None)
            vals = vals[:len(fields)]
        
        row = {fields[i]: _parse_value(vals[i]) for i in range(len(fields))}
        res.append(row)
        
    if strict and len(res) != expected:
        raise ToonError(f"Expected {expected} rows, got {len(res)}")
    return res


def _parse_list_array(lines: List[str], indent: int, strict: bool) -> list:
    res = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped: # Skip blank
            i += 1
            continue
            
        if stripped.startswith("-"):
            content = stripped[1:].strip()
            
            # Check for Objects as List Items (§10)
            # Case 1: "- key[N]..." -> Tabular array on hyphen line
            if _is_header(content):
                # This is a list object starting with a tabular array
                # Decode that tabular array using subsequent lines
                # But where do subsequent fields go?
                # Spec: "Tabular rows MUST appear at depth +2... All other fields... at depth +1"
                # This complex parsing requires identifying where the object ends.
                # Simplified approach for v3.0 logic:
                # We need to parse a dictionary here.
                # Let's delegate to a helper that parses 'Object starting at line i'
                pass # Logic below handles this via recursion or structure? 
                
                # Actually, - header indicates the start of an object whose first field is that array.
                # We can treat the "- " as indentation level for a new parsing context.
                # But _decode_object expects "key: value" lines.
                # Here we have "- key[N]..."
                
                # Let's extract key and value from the line content
                # It's a header line, so it defines a key and a value (the array).
                # We treat this as the first line of an object block.
                
                # Find the object block lines
                # The object is indented at the level of the hyphen + space? 
                # No, standard fields are at +1 depth relative to hyphen.
                pass 

            # Case: Standard "- key: val" or "- primitive"
            if ":" in content and not _is_header(content):
                 # Likely object first field
                 # Check strict "key: val" syntax
                 col = content.index(":")
                 k = content[:col].strip()
                 v_part = content[col+1:].strip()
                 
                 obj = {}
                 if v_part:
                     obj[k] = _parse_value(v_part)
                     # Check next lines for siblings
                     # Siblings at same indent as "-" (Wait, relative to hyphen?)
                     # List items:
                     # - key: val
                     #   key2: val
                     # Sibling key2 is aligned with key.
                 else:
                     # Nested value
                     # Parse children
                     pass
                 
                 # NOTE: Providing a robust implementation of §10 list-item parsing 
                 # requires lookahead and indentation tracking that fits the 
                 # `_decode_object` model.
                 # For now, we will use a simpler approximation that works for 
                 # standard cases produced by our encoder.
                 
            # Fallback simplistic list parsing (matches v1.3 behavior + improvements)
            val_str = content
            if not val_str:
                res.append({}) # Empty object "-"
            elif ":" in val_str and not val_str.startswith("["): # Avoid headers
                # Inline object property "- key: val"
                # This is actually the first line of an object.
                # We need to capture subsequent lines indented to this level.
                
                # Find all lines belonging to this item
                item_lines = [content] # The part after "- " acts as first line
                start_ind = len(line) - len(line.lstrip())
                # Sibling fields are indented by 2 chars usually? 
                # - key: val
                #   key2: val
                # Indent of key2 is start_ind + 2
                
                # Consume validation
                j = i + 1
                while j < len(lines):
                    next_l = lines[j]
                    if not next_l.strip(): 
                        j += 1
                        continue
                    next_ind = len(next_l) - len(next_l.lstrip())
                    if next_ind <= start_ind: # Back to list item level
                        break
                    item_lines.append(next_l)
                    j += 1
                
                # Decode object from these lines
                # Prepend dummy indentation to first line to match others?
                # Actually _decode_object handles indent stripping.
                # We just pass lines.
                obj, _ = _decode_object(item_lines, 0, indent, strict)
                res.append(obj)
                i = j
                continue

            else:
                res.append(_parse_value(val_str))
        i += 1
    return res

def _clean_input(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def _split_row(line, delimiter):
    values = []
    current = io.StringIO()
    in_quotes = False
    escape = False
    i = 0
    while i < len(line):
        c = line[i]
        if escape:
            current.write(c); escape=False; i+=1; continue
        if c == '\\' and in_quotes:
            escape=True; current.write(c); i+=1; continue
        if c == '"':
            in_quotes = not in_quotes; current.write(c); i+=1; continue
        if c == delimiter and not in_quotes:
            values.append(current.getvalue()); current = io.StringIO(); i+=1; continue
        current.write(c); i+=1
    values.append(current.getvalue())
    return values

def _parse_value(val):
    val = val.strip()
    if not val: return None
    if val == "true": return True
    if val == "false": return False
    if val == "null": return None
    if val.startswith('"') and val.endswith('"'): return _unescape_string(val)
    if re.match(r"^-?\d+$", val): return int(val)
    if re.match(r"^-?\d+\.\d+$", val): return float(val)
    return val

def _unescape_string(val):
    s = val[1:-1]
    out = io.StringIO()
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and i+1 < len(s):
            nc = s[i+1]
            if nc=='n': out.write('\n')
            elif nc=='r': out.write('\r')
            elif nc=='t': out.write('\t')
            elif nc=='"': out.write('"')
            elif nc=='\\': out.write('\\')
            else: out.write(c); i+=1; continue
            i+=2
        else: out.write(c); i+=1
    return out.getvalue()

def _parse_inline_values(text, delimiter):
    return [_parse_value(v) for v in _split_row(text.strip(), delimiter)]

# Convenience
def from_json(json_str, **kwargs):
    import json
    return encode(json.loads(json_str), **kwargs)

def to_json(toon_str, indent=2, **kwargs):
    import json
    return json.dumps(decode(toon_str, **kwargs), ensure_ascii=False, indent=indent)

def encode_for_llm(data, delimiter="\t"):
    return encode(data, delimiter=delimiter)

def wrap_for_prompt(toon_str):
    return f"```toon\n{toon_str}\n```"
