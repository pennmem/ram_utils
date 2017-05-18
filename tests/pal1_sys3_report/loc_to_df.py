import xlrd
import sys
import pandas as pd


def loc_to_df(excelFile):
    workbook = xlrd.open_workbook(excelFile)
    sheetNumbers = []

    if isStereo(workbook):
        MTL_sheet = findInSheetName(workbook, 'MTL')
        if not MTL_sheet==None:
            sheetNumbers.append(MTL_sheet)
        else:
            raise BadExcel(excelFile)
    elif isDepth(workbook):
        depth_sheet = findInSheetName(workbook, 'depth')
        if (not depth_sheet==None) or (depth_sheet==0):
            sheetNumbers.append(depth_sheet)
        else:
            raise BadExcel(excelFile)

    surface_sheet = findInSheetName(workbook, 'surface')
    if not surface_sheet==None:
        sheetNumbers.append(surface_sheet)

    loc = []
    for sheet_i, sheetNumber in enumerate(sheetNumbers):
        worksheet = workbook.sheet_by_index(sheetNumber)
        num_rows = worksheet.nrows - 1
        curr_row = -1 if sheet_i == 0 else 0
        #maxLen = -1
        while curr_row < num_rows:
            r = []
            curr_row += 1
            row = worksheet.row(curr_row)
            if len(row)==0  or all([len(str(cell.value))==0 for cell in row]):
                continue
            if 'COMMENTS' in str(row[0].value).upper() \
                    or 'NEURORADIOLOGIST' in str(row[0].value).upper()\
                    or 'COMPLETED' in str(row[0].value).upper():
                break
            #rowLen = 0;
            for cell in row:
                #rowLen+=1;
                #f.write(str(cell.value)+'\t')
                r.append(str(cell.value).replace('_','\\textunderscore'))
            #if rowLen>maxLen:
            #    maxLen = rowLen
            #f.write('\t'*(maxLen-rowLen))
            loc.append(r)
    return pd.DataFrame([r[1:4] for r in loc[1:]], index=[r[0].upper() for r in loc[1:]], columns=loc[0][1:4])

def isStereo(workbook):
    stereoSheet = findInSheetName(workbook, 'STEREO')
    return (stereoSheet != None)
       
def isDepth(workbook):
    depthSheet = findInSheetName(workbook, 'DEPTH')
    return (depthSheet != None)

def findInSheetName(workbook, strToFind):
    worksheet_names = workbook.sheet_names()
    for i, name in enumerate(worksheet_names):
        if strToFind.upper() in name.upper():
            return i
    return None

class BadExcel(Exception):
    def __init__(self, filename):
        self.filename = filename

    def __str__(self):
        return 'NO DEPTH OR MTL: %s'%(self.filename)
