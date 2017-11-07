import os
import re

def replace_template(template_file_name, out_file_name, replace_dict={}):
    """ String substitution using a template file and a dictionary containing values to be replaced

    template_file_name:  str
        Template to load
    out_file_name: str
        Desired output file name
    replace_dict: dict
        Keys are template placeholders and values are the data that should be substituted into the template

    """
    in_file = open(template_file_name,'r')
    out_file = open(out_file_name,'w')

    new_line_tmp = 
    out_file.write(new_line_tmp + "\n")

    in_file.close()
    out_file.close()
    return

def replace_template_to_string(template_file_name,replace_dict={}):
    """ String substitution using a template file

    template_file_name: str
        Template to use
    replace_dict: dict
        Keys are template placeholders and values are the data that should be substituted

    Returns
    -------
    out_str: str
        String containing the template with substituted values
    """

    out_str = ''
    in_file = open(template_file_name,'r')

    for line in in_file.readlines():
        line = line.rstrip()
        new_line_tmp = str(line)
        for name_in_template, replace_val in replace_dict.iteritems():
            # using lambda function in the replace string to make sure that escaped characters such as \n \r \b are
            # not being interpreted but treated literarly
            repl_fcn = lambda x:str(replace_val)
            new_line_tmp = re.sub(pattern=name_in_template, repl=repl_fcn, string=new_line_tmp)
            new_line_tmp = new_line_tmp.rstrip()
        out_str += new_line_tmp+'\n'

    in_file.close()
    return out_str

def _template_substitution_generator(in_file, replace_dict={}):
    for line in in_file.readline():
        line = line.rstrip()
        new_line_tmp = str(line)
        for name_in_template, replace_val in replace_dict.iteritems():
            # using lambda function in the replace string to make sure that escaped characters such as \n \r \b are
            # not being interpreted but treated literarly
            repl_fcn = lambda x: str(replace_val)
            new_line_tmp = re.sub(pattern=name_in_template, repl=repl_fcn, string=new_line_tmp)
            new_line_tmp = new_line_tmp.rstrip()
        yield new_line_temp


