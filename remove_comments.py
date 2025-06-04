import tokenize
import io

def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    output_tokens = []
    prev_toktype = None
    first_token = True

    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok.type
        token_string = tok.string

        # Remove comments
        if token_type == tokenize.COMMENT:
            continue

        # Remove standalone (unassigned) docstrings
        if token_type == tokenize.STRING:
            if prev_toktype == tokenize.INDENT or first_token:
                first_token = False
                continue

        output_tokens.append((token_type, token_string))
        prev_toktype = token_type
        first_token = False

    # Reconstruct code
    cleaned_code = tokenize.untokenize(output_tokens)

    # Post-process: collapse multiple consecutive empty lines to one
    lines = cleaned_code.splitlines()
    new_lines = []
    prev_empty = False
    for line in lines:
        if line.strip() == "":
            if not prev_empty:
                new_lines.append("")
            prev_empty = True
        else:
            new_lines.append(line)
            prev_empty = False
    return "\n".join(new_lines) + "\n"

# File I/O
input_file = 'secourses_app.py'
output_file = 'secourses_app.py'

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        source = f.read()
    cleaned = remove_comments_and_docstrings(source)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    print(f"Comments and docstrings removed. Output saved as {output_file}")
except Exception as e:
    print(f"Error: {e}")