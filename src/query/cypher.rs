//! Cypher Parser and Transpiler
//!
//! Parses Cypher queries and transpiles them to SQL with recursive CTEs.
//! This enables graph queries over the relational Lance storage.
//!
//! # Supported Cypher Features
//!
//! ```cypher
//! -- Simple pattern matching
//! MATCH (a:Thought)-[:CAUSES]->(b:Thought)
//! WHERE a.qidx > 100
//! RETURN b
//!
//! -- Variable-length paths (recursive CTE)
//! MATCH (a)-[:CAUSES*1..5]->(b)
//! WHERE a.id = 'start'
//! RETURN b, path, amplification
//!
//! -- Multiple relationships
//! MATCH (a)-[:CAUSES|ENABLES]->(b)
//! RETURN a, b
//!
//! -- Create operations
//! CREATE (a:Thought {content: 'Hello'})
//! CREATE (a)-[:CAUSES {weight: 0.8}]->(b)
//! ```

use std::collections::HashMap;
use crate::{Error, Result};

// =============================================================================
// AST TYPES
// =============================================================================

/// Parsed Cypher query
#[derive(Debug, Clone)]
pub struct CypherQuery {
    pub query_type: QueryType,
    pub match_clause: Option<MatchClause>,
    pub where_clause: Option<WhereClause>,
    pub return_clause: Option<ReturnClause>,
    pub order_by: Option<OrderByClause>,
    pub limit: Option<u64>,
    pub skip: Option<u64>,
    pub create_clause: Option<CreateClause>,
    pub set_clause: Option<SetClause>,
    pub delete_clause: Option<DeleteClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    Match,
    Create,
    Merge,
    Delete,
    Set,
}

/// MATCH clause: pattern to search for
#[derive(Debug, Clone)]
pub struct MatchClause {
    pub patterns: Vec<Pattern>,
}

/// A graph pattern: (node)-[edge]->(node)...
#[derive(Debug, Clone)]
pub struct Pattern {
    pub elements: Vec<PatternElement>,
}

#[derive(Debug, Clone)]
pub enum PatternElement {
    Node(NodePattern),
    Edge(EdgePattern),
}

/// Node pattern: (alias:Label {props})
#[derive(Debug, Clone)]
pub struct NodePattern {
    pub alias: Option<String>,
    pub labels: Vec<String>,
    pub properties: HashMap<String, Value>,
}

/// Edge pattern: -[alias:TYPE*min..max {props}]->
#[derive(Debug, Clone)]
pub struct EdgePattern {
    pub alias: Option<String>,
    pub types: Vec<String>,
    pub direction: EdgeDirection,
    pub min_hops: u32,
    pub max_hops: u32,
    pub properties: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EdgeDirection {
    Outgoing,  // ->
    Incoming,  // <-
    Both,      // -
}

/// WHERE clause conditions
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub condition: Condition,
}

#[derive(Debug, Clone)]
pub enum Condition {
    Comparison {
        left: Expr,
        op: ComparisonOp,
        right: Expr,
    },
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    Not(Box<Condition>),
    IsNull(Expr),
    IsNotNull(Expr),
    In(Expr, Vec<Value>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOp {
    Eq,      // =
    Ne,      // <>
    Lt,      // <
    Le,      // <=
    Gt,      // >
    Ge,      // >=
    Contains,
    StartsWith,
    EndsWith,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Property { alias: String, property: String },
    Literal(Value),
    Function { name: String, args: Vec<Expr> },
    Variable(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Value {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
    List(Vec<Value>),
}

/// RETURN clause
#[derive(Debug, Clone)]
pub struct ReturnClause {
    pub items: Vec<ReturnItem>,
    pub distinct: bool,
}

#[derive(Debug, Clone)]
pub struct ReturnItem {
    pub expr: Expr,
    pub alias: Option<String>,
}

/// ORDER BY clause
#[derive(Debug, Clone)]
pub struct OrderByClause {
    pub items: Vec<OrderItem>,
}

#[derive(Debug, Clone)]
pub struct OrderItem {
    pub expr: Expr,
    pub direction: SortDirection,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// CREATE clause
#[derive(Debug, Clone)]
pub struct CreateClause {
    pub patterns: Vec<Pattern>,
}

/// SET clause
#[derive(Debug, Clone)]
pub struct SetClause {
    pub items: Vec<SetItem>,
}

#[derive(Debug, Clone)]
pub struct SetItem {
    pub target: Expr,
    pub value: Expr,
}

/// DELETE clause
#[derive(Debug, Clone)]
pub struct DeleteClause {
    pub items: Vec<String>,
    pub detach: bool,
}

// =============================================================================
// PARSER
// =============================================================================

/// Cypher parser
pub struct CypherParser {
    tokens: Vec<Token>,
    pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords
    Match,
    Where,
    Return,
    Create,
    Merge,
    Delete,
    Detach,
    Set,
    OrderBy,
    Limit,
    Skip,
    And,
    Or,
    Not,
    In,
    Is,
    Null,
    Distinct,
    As,
    Asc,
    Desc,
    Contains,
    StartsWith,
    EndsWith,
    
    // Symbols
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Colon,
    Comma,
    Dot,
    Pipe,
    Star,
    DotDot,
    Arrow,        // ->
    LeftArrow,    // <-
    Dash,         // -
    
    // Operators
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    
    // Literals
    Identifier(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    
    // End
    Eof,
}

impl CypherParser {
    /// Parse a Cypher query string
    pub fn parse(input: &str) -> Result<CypherQuery> {
        let tokens = Self::tokenize(input)?;
        let mut parser = Self { tokens, pos: 0 };
        parser.parse_query()
    }
    
    /// Tokenize input string
    fn tokenize(input: &str) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            let c = chars[i];
            
            // Skip whitespace
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            
            // Skip comments
            if c == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            
            // Symbols
            match c {
                '(' => { tokens.push(Token::LParen); i += 1; continue; }
                ')' => { tokens.push(Token::RParen); i += 1; continue; }
                '[' => { tokens.push(Token::LBracket); i += 1; continue; }
                ']' => { tokens.push(Token::RBracket); i += 1; continue; }
                '{' => { tokens.push(Token::LBrace); i += 1; continue; }
                '}' => { tokens.push(Token::RBrace); i += 1; continue; }
                ':' => { tokens.push(Token::Colon); i += 1; continue; }
                ',' => { tokens.push(Token::Comma); i += 1; continue; }
                '|' => { tokens.push(Token::Pipe); i += 1; continue; }
                '*' => { tokens.push(Token::Star); i += 1; continue; }
                '=' => { tokens.push(Token::Eq); i += 1; continue; }
                _ => {}
            }
            
            // Multi-char operators
            if c == '-' {
                if i + 1 < chars.len() && chars[i + 1] == '>' {
                    tokens.push(Token::Arrow);
                    i += 2;
                    continue;
                } else {
                    tokens.push(Token::Dash);
                    i += 1;
                    continue;
                }
            }
            
            if c == '<' {
                if i + 1 < chars.len() {
                    match chars[i + 1] {
                        '-' => { tokens.push(Token::LeftArrow); i += 2; continue; }
                        '=' => { tokens.push(Token::Le); i += 2; continue; }
                        '>' => { tokens.push(Token::Ne); i += 2; continue; }
                        _ => { tokens.push(Token::Lt); i += 1; continue; }
                    }
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                    continue;
                }
            }
            
            if c == '>' {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Ge);
                    i += 2;
                    continue;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                    continue;
                }
            }
            
            if c == '.' {
                if i + 1 < chars.len() && chars[i + 1] == '.' {
                    tokens.push(Token::DotDot);
                    i += 2;
                    continue;
                } else {
                    tokens.push(Token::Dot);
                    i += 1;
                    continue;
                }
            }
            
            // String literals
            if c == '\'' || c == '"' {
                let quote = c;
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != quote {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::StringLit(s));
                i += 1; // skip closing quote
                continue;
            }
            
            // Numbers
            if c.is_ascii_digit() || (c == '-' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
                let start = i;
                if c == '-' { i += 1; }
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                if num_str.contains('.') {
                    tokens.push(Token::FloatLit(num_str.parse().unwrap()));
                } else {
                    tokens.push(Token::IntLit(num_str.parse().unwrap()));
                }
                continue;
            }
            
            // Identifiers and keywords
            if c.is_alphabetic() || c == '_' {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let token = match word.to_uppercase().as_str() {
                    "MATCH" => Token::Match,
                    "WHERE" => Token::Where,
                    "RETURN" => Token::Return,
                    "CREATE" => Token::Create,
                    "MERGE" => Token::Merge,
                    "DELETE" => Token::Delete,
                    "DETACH" => Token::Detach,
                    "SET" => Token::Set,
                    "ORDER" => {
                        // Check for ORDER BY
                        while i < chars.len() && chars[i].is_whitespace() { i += 1; }
                        if i + 1 < chars.len() {
                            let by_start = i;
                            while i < chars.len() && chars[i].is_alphabetic() { i += 1; }
                            let by_word: String = chars[by_start..i].iter().collect();
                            if by_word.to_uppercase() == "BY" {
                                Token::OrderBy
                            } else {
                                i = by_start; // reset
                                Token::Identifier(word)
                            }
                        } else {
                            Token::Identifier(word)
                        }
                    }
                    "BY" => Token::Identifier(word), // handled in ORDER
                    "LIMIT" => Token::Limit,
                    "SKIP" => Token::Skip,
                    "AND" => Token::And,
                    "OR" => Token::Or,
                    "NOT" => Token::Not,
                    "IN" => Token::In,
                    "IS" => Token::Is,
                    "NULL" => Token::Null,
                    "DISTINCT" => Token::Distinct,
                    "AS" => Token::As,
                    "ASC" => Token::Asc,
                    "DESC" => Token::Desc,
                    "CONTAINS" => Token::Contains,
                    "STARTS" => {
                        // STARTS WITH
                        while i < chars.len() && chars[i].is_whitespace() { i += 1; }
                        let with_start = i;
                        while i < chars.len() && chars[i].is_alphabetic() { i += 1; }
                        let with_word: String = chars[with_start..i].iter().collect();
                        if with_word.to_uppercase() == "WITH" {
                            Token::StartsWith
                        } else {
                            i = with_start;
                            Token::Identifier(word)
                        }
                    }
                    "ENDS" => {
                        // ENDS WITH
                        while i < chars.len() && chars[i].is_whitespace() { i += 1; }
                        let with_start = i;
                        while i < chars.len() && chars[i].is_alphabetic() { i += 1; }
                        let with_word: String = chars[with_start..i].iter().collect();
                        if with_word.to_uppercase() == "WITH" {
                            Token::EndsWith
                        } else {
                            i = with_start;
                            Token::Identifier(word)
                        }
                    }
                    "TRUE" => Token::BoolLit(true),
                    "FALSE" => Token::BoolLit(false),
                    _ => Token::Identifier(word),
                };
                tokens.push(token);
                continue;
            }
            
            return Err(Error::Query(format!("Unexpected character: {}", c)));
        }
        
        tokens.push(Token::Eof);
        Ok(tokens)
    }
    
    fn current(&self) -> &Token {
        &self.tokens[self.pos]
    }
    
    fn advance(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }
    
    fn expect(&mut self, expected: Token) -> Result<()> {
        if std::mem::discriminant(self.current()) == std::mem::discriminant(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(Error::Query(format!(
                "Expected {:?}, got {:?}", expected, self.current()
            )))
        }
    }
    
    fn parse_query(&mut self) -> Result<CypherQuery> {
        let mut query = CypherQuery {
            query_type: QueryType::Match,
            match_clause: None,
            where_clause: None,
            return_clause: None,
            order_by: None,
            limit: None,
            skip: None,
            create_clause: None,
            set_clause: None,
            delete_clause: None,
        };
        
        match self.current() {
            Token::Match => {
                query.query_type = QueryType::Match;
                self.advance();
                query.match_clause = Some(self.parse_match()?);
            }
            Token::Create => {
                query.query_type = QueryType::Create;
                self.advance();
                query.create_clause = Some(self.parse_create()?);
            }
            _ => return Err(Error::Query("Expected MATCH or CREATE".into())),
        }
        
        // Optional WHERE
        if matches!(self.current(), Token::Where) {
            self.advance();
            query.where_clause = Some(self.parse_where()?);
        }
        
        // Optional RETURN
        if matches!(self.current(), Token::Return) {
            self.advance();
            query.return_clause = Some(self.parse_return()?);
        }
        
        // Optional ORDER BY
        if matches!(self.current(), Token::OrderBy) {
            self.advance();
            query.order_by = Some(self.parse_order_by()?);
        }
        
        // Optional LIMIT
        if matches!(self.current(), Token::Limit) {
            self.advance();
            if let Token::IntLit(n) = self.advance() {
                query.limit = Some(n as u64);
            }
        }
        
        // Optional SKIP
        if matches!(self.current(), Token::Skip) {
            self.advance();
            if let Token::IntLit(n) = self.advance() {
                query.skip = Some(n as u64);
            }
        }
        
        Ok(query)
    }
    
    fn parse_match(&mut self) -> Result<MatchClause> {
        let patterns = vec![self.parse_pattern()?];
        Ok(MatchClause { patterns })
    }
    
    fn parse_pattern(&mut self) -> Result<Pattern> {
        let mut elements = Vec::new();
        
        // First element must be a node
        elements.push(PatternElement::Node(self.parse_node_pattern()?));
        
        // Then alternating edges and nodes
        loop {
            if self.is_edge_start() {
                elements.push(PatternElement::Edge(self.parse_edge_pattern()?));
                elements.push(PatternElement::Node(self.parse_node_pattern()?));
            } else {
                break;
            }
        }
        
        Ok(Pattern { elements })
    }
    
    fn is_edge_start(&self) -> bool {
        matches!(self.current(), Token::Dash | Token::LeftArrow)
    }
    
    fn parse_node_pattern(&mut self) -> Result<NodePattern> {
        self.expect(Token::LParen)?;
        
        let mut node = NodePattern {
            alias: None,
            labels: Vec::new(),
            properties: HashMap::new(),
        };
        
        // Optional alias
        if let Token::Identifier(id) = self.current() {
            node.alias = Some(id.clone());
            self.advance();
        }
        
        // Optional labels
        while matches!(self.current(), Token::Colon) {
            self.advance();
            if let Token::Identifier(label) = self.advance() {
                node.labels.push(label);
            }
        }
        
        // Optional properties
        if matches!(self.current(), Token::LBrace) {
            node.properties = self.parse_properties()?;
        }
        
        self.expect(Token::RParen)?;
        Ok(node)
    }
    
    fn parse_edge_pattern(&mut self) -> Result<EdgePattern> {
        let mut edge = EdgePattern {
            alias: None,
            types: Vec::new(),
            direction: EdgeDirection::Outgoing,
            min_hops: 1,
            max_hops: 1,
            properties: HashMap::new(),
        };
        
        // Direction start
        if matches!(self.current(), Token::LeftArrow) {
            edge.direction = EdgeDirection::Incoming;
            self.advance();
        } else {
            self.expect(Token::Dash)?;
        }
        
        // Edge details [...]
        if matches!(self.current(), Token::LBracket) {
            self.advance();
            
            // Optional alias
            if let Token::Identifier(id) = self.current() {
                edge.alias = Some(id.clone());
                self.advance();
            }
            
            // Optional types
            while matches!(self.current(), Token::Colon | Token::Pipe) {
                if matches!(self.current(), Token::Pipe) {
                    self.advance();
                } else {
                    self.advance(); // colon
                }
                if let Token::Identifier(t) = self.advance() {
                    edge.types.push(t);
                }
            }
            
            // Optional variable length *min..max
            if matches!(self.current(), Token::Star) {
                self.advance();
                
                // min
                if let Token::IntLit(n) = self.current() {
                    edge.min_hops = *n as u32;
                    self.advance();
                }
                
                // ..max
                if matches!(self.current(), Token::DotDot) {
                    self.advance();
                    if let Token::IntLit(n) = self.current() {
                        edge.max_hops = *n as u32;
                        self.advance();
                    } else {
                        edge.max_hops = 10; // default max
                    }
                } else {
                    edge.max_hops = edge.min_hops;
                }
            }
            
            // Optional properties
            if matches!(self.current(), Token::LBrace) {
                edge.properties = self.parse_properties()?;
            }
            
            self.expect(Token::RBracket)?;
        }
        
        // Direction end
        if edge.direction == EdgeDirection::Incoming {
            self.expect(Token::Dash)?;
        } else if matches!(self.current(), Token::Arrow) {
            self.advance();
        } else {
            self.expect(Token::Dash)?;
            edge.direction = EdgeDirection::Both;
        }
        
        Ok(edge)
    }
    
    fn parse_properties(&mut self) -> Result<HashMap<String, Value>> {
        self.expect(Token::LBrace)?;
        let mut props = HashMap::new();
        
        loop {
            if matches!(self.current(), Token::RBrace) {
                break;
            }
            
            // key: value
            let key = if let Token::Identifier(k) = self.advance() {
                k
            } else {
                return Err(Error::Query("Expected property key".into()));
            };
            
            self.expect(Token::Colon)?;
            let value = self.parse_value()?;
            props.insert(key, value);
            
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.expect(Token::RBrace)?;
        Ok(props)
    }
    
    fn parse_value(&mut self) -> Result<Value> {
        match self.advance() {
            Token::StringLit(s) => Ok(Value::String(s)),
            Token::IntLit(n) => Ok(Value::Integer(n)),
            Token::FloatLit(f) => Ok(Value::Float(f)),
            Token::BoolLit(b) => Ok(Value::Boolean(b)),
            Token::Null => Ok(Value::Null),
            t => Err(Error::Query(format!("Expected value, got {:?}", t))),
        }
    }
    
    fn parse_where(&mut self) -> Result<WhereClause> {
        let condition = self.parse_condition()?;
        Ok(WhereClause { condition })
    }
    
    fn parse_condition(&mut self) -> Result<Condition> {
        let mut left = self.parse_comparison()?;
        
        loop {
            match self.current() {
                Token::And => {
                    self.advance();
                    let right = self.parse_comparison()?;
                    left = Condition::And(Box::new(left), Box::new(right));
                }
                Token::Or => {
                    self.advance();
                    let right = self.parse_comparison()?;
                    left = Condition::Or(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        
        Ok(left)
    }
    
    fn parse_comparison(&mut self) -> Result<Condition> {
        let left = self.parse_expr()?;
        
        let op = match self.current() {
            Token::Eq => ComparisonOp::Eq,
            Token::Ne => ComparisonOp::Ne,
            Token::Lt => ComparisonOp::Lt,
            Token::Le => ComparisonOp::Le,
            Token::Gt => ComparisonOp::Gt,
            Token::Ge => ComparisonOp::Ge,
            Token::Contains => ComparisonOp::Contains,
            Token::StartsWith => ComparisonOp::StartsWith,
            Token::EndsWith => ComparisonOp::EndsWith,
            Token::Is => {
                self.advance();
                if matches!(self.current(), Token::Not) {
                    self.advance();
                    self.expect(Token::Null)?;
                    return Ok(Condition::IsNotNull(left));
                } else {
                    self.expect(Token::Null)?;
                    return Ok(Condition::IsNull(left));
                }
            }
            _ => return Ok(Condition::Comparison {
                left: left.clone(),
                op: ComparisonOp::Eq,
                right: Expr::Literal(Value::Boolean(true)),
            }),
        };
        
        self.advance();
        let right = self.parse_expr()?;
        
        Ok(Condition::Comparison { left, op, right })
    }
    
    fn parse_expr(&mut self) -> Result<Expr> {
        match self.current().clone() {
            Token::Identifier(name) => {
                self.advance();
                if matches!(self.current(), Token::Dot) {
                    self.advance();
                    if let Token::Identifier(prop) = self.advance() {
                        Ok(Expr::Property { alias: name, property: prop })
                    } else {
                        Err(Error::Query("Expected property name".into()))
                    }
                } else if matches!(self.current(), Token::LParen) {
                    // Function call
                    self.advance();
                    let mut args = Vec::new();
                    while !matches!(self.current(), Token::RParen) {
                        args.push(self.parse_expr()?);
                        if matches!(self.current(), Token::Comma) {
                            self.advance();
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Expr::Function { name, args })
                } else {
                    Ok(Expr::Variable(name))
                }
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(Expr::Literal(Value::String(s)))
            }
            Token::IntLit(n) => {
                self.advance();
                Ok(Expr::Literal(Value::Integer(n)))
            }
            Token::FloatLit(f) => {
                self.advance();
                Ok(Expr::Literal(Value::Float(f)))
            }
            Token::BoolLit(b) => {
                self.advance();
                Ok(Expr::Literal(Value::Boolean(b)))
            }
            Token::Null => {
                self.advance();
                Ok(Expr::Literal(Value::Null))
            }
            _ => Err(Error::Query(format!("Unexpected token in expression: {:?}", self.current()))),
        }
    }
    
    fn parse_return(&mut self) -> Result<ReturnClause> {
        let distinct = if matches!(self.current(), Token::Distinct) {
            self.advance();
            true
        } else {
            false
        };
        
        let mut items = Vec::new();
        loop {
            let expr = self.parse_expr()?;
            let alias = if matches!(self.current(), Token::As) {
                self.advance();
                if let Token::Identifier(a) = self.advance() {
                    Some(a)
                } else {
                    None
                }
            } else {
                None
            };
            items.push(ReturnItem { expr, alias });
            
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        Ok(ReturnClause { items, distinct })
    }
    
    fn parse_order_by(&mut self) -> Result<OrderByClause> {
        let mut items = Vec::new();
        
        loop {
            let expr = self.parse_expr()?;
            let direction = match self.current() {
                Token::Desc => { self.advance(); SortDirection::Desc }
                Token::Asc => { self.advance(); SortDirection::Asc }
                _ => SortDirection::Asc,
            };
            items.push(OrderItem { expr, direction });
            
            if matches!(self.current(), Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        Ok(OrderByClause { items })
    }
    
    fn parse_create(&mut self) -> Result<CreateClause> {
        let patterns = vec![self.parse_pattern()?];
        Ok(CreateClause { patterns })
    }
}

// =============================================================================
// TRANSPILER (Cypher → SQL)
// =============================================================================

/// Transpile Cypher AST to SQL
pub struct CypherTranspiler;

impl CypherTranspiler {
    /// Transpile a Cypher query to SQL
    pub fn transpile(query: &CypherQuery) -> Result<String> {
        match query.query_type {
            QueryType::Match => Self::transpile_match(query),
            QueryType::Create => Self::transpile_create(query),
            _ => Err(Error::Query("Unsupported query type".into())),
        }
    }
    
    fn transpile_match(query: &CypherQuery) -> Result<String> {
        let match_clause = query.match_clause.as_ref()
            .ok_or_else(|| Error::Query("Missing MATCH clause".into()))?;
        
        let pattern = &match_clause.patterns[0];
        
        // Determine if we need recursive CTE
        let needs_recursive = pattern.elements.iter().any(|e| {
            if let PatternElement::Edge(edge) = e {
                edge.max_hops > 1
            } else {
                false
            }
        });
        
        if needs_recursive {
            Self::transpile_recursive_match(query, pattern)
        } else {
            Self::transpile_simple_match(query, pattern)
        }
    }
    
    fn transpile_simple_match(query: &CypherQuery, pattern: &Pattern) -> Result<String> {
        let mut sql = String::new();
        let mut tables = Vec::new();
        let mut joins = Vec::new();
        let mut where_parts = Vec::new();
        
        let mut node_idx = 0;
        let mut edge_idx = 0;
        
        for element in &pattern.elements {
            match element {
                PatternElement::Node(node) => {
                    let alias = node.alias.clone()
                        .unwrap_or_else(|| format!("n{}", node_idx));
                    
                    if node_idx == 0 {
                        tables.push(format!("nodes AS {}", alias));
                    }
                    
                    // Label filter
                    if !node.labels.is_empty() {
                        where_parts.push(format!(
                            "{}.label = '{}'",
                            alias,
                            node.labels[0]
                        ));
                    }
                    
                    // Property filters
                    for (key, value) in &node.properties {
                        where_parts.push(format!(
                            "{}.{} = {}",
                            alias,
                            key,
                            Self::value_to_sql(value)
                        ));
                    }
                    
                    node_idx += 1;
                }
                PatternElement::Edge(edge) => {
                    let edge_alias = edge.alias.clone()
                        .unwrap_or_else(|| format!("e{}", edge_idx));
                    let prev_node_alias = pattern.elements.get(node_idx * 2 - 2)
                        .and_then(|e| if let PatternElement::Node(n) = e { n.alias.clone() } else { None })
                        .unwrap_or_else(|| format!("n{}", node_idx - 1));
                    let next_node_alias = format!("n{}", node_idx);
                    
                    // Join edge table
                    let (from_col, to_col) = match edge.direction {
                        EdgeDirection::Outgoing => ("from_id", "to_id"),
                        EdgeDirection::Incoming => ("to_id", "from_id"),
                        EdgeDirection::Both => ("from_id", "to_id"), // simplified
                    };
                    
                    joins.push(format!(
                        "JOIN edges AS {} ON {}.id = {}.{}",
                        edge_alias, prev_node_alias, edge_alias, from_col
                    ));
                    joins.push(format!(
                        "JOIN nodes AS {} ON {}.{} = {}.id",
                        next_node_alias, edge_alias, to_col, next_node_alias
                    ));
                    
                    // Edge type filter
                    if !edge.types.is_empty() {
                        let types_sql = edge.types.iter()
                            .map(|t| format!("'{}'", t))
                            .collect::<Vec<_>>()
                            .join(", ");
                        where_parts.push(format!("{}.type IN ({})", edge_alias, types_sql));
                    }
                    
                    edge_idx += 1;
                }
            }
        }
        
        // Build SELECT clause
        let select_cols = if let Some(ref ret) = query.return_clause {
            ret.items.iter()
                .map(|item| Self::expr_to_sql(&item.expr))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            "*".to_string()
        };
        
        sql.push_str(&format!("SELECT {}\n", select_cols));
        sql.push_str(&format!("FROM {}\n", tables.join(", ")));
        
        for join in joins {
            sql.push_str(&join);
            sql.push('\n');
        }
        
        // WHERE clause
        if let Some(ref where_clause) = query.where_clause {
            where_parts.push(Self::condition_to_sql(&where_clause.condition));
        }
        
        if !where_parts.is_empty() {
            sql.push_str(&format!("WHERE {}\n", where_parts.join(" AND ")));
        }
        
        // ORDER BY
        if let Some(ref order) = query.order_by {
            let order_sql = order.items.iter()
                .map(|item| {
                    let dir = if item.direction == SortDirection::Desc { "DESC" } else { "ASC" };
                    format!("{} {}", Self::expr_to_sql(&item.expr), dir)
                })
                .collect::<Vec<_>>()
                .join(", ");
            sql.push_str(&format!("ORDER BY {}\n", order_sql));
        }
        
        // LIMIT
        if let Some(limit) = query.limit {
            sql.push_str(&format!("LIMIT {}\n", limit));
        }
        
        // OFFSET (SKIP)
        if let Some(skip) = query.skip {
            sql.push_str(&format!("OFFSET {}\n", skip));
        }
        
        Ok(sql)
    }
    
    fn transpile_recursive_match(query: &CypherQuery, pattern: &Pattern) -> Result<String> {
        // Extract start node, edge, and end node
        let (start_node, edge, end_node) = Self::extract_path_pattern(pattern)?;
        
        let edge_type_filter = if !edge.types.is_empty() {
            format!(
                "AND e.type IN ({})",
                edge.types.iter().map(|t| format!("'{}'", t)).collect::<Vec<_>>().join(", ")
            )
        } else {
            String::new()
        };
        
        // Start condition
        let start_where = if !start_node.labels.is_empty() {
            format!("WHERE label = '{}'", start_node.labels[0])
        } else {
            String::new()
        };
        
        // Build recursive CTE
        let sql = format!(r#"
WITH RECURSIVE traverse AS (
    -- Base case: start nodes
    SELECT 
        id,
        ARRAY[id] as path,
        1.0 as amplification,
        0 as depth
    FROM nodes
    {start_where}
    
    UNION ALL
    
    -- Recursive case: follow edges
    SELECT 
        n.id,
        t.path || n.id,
        t.amplification * COALESCE(e.amplification, e.weight, 1.0),
        t.depth + 1
    FROM traverse t
    JOIN edges e ON t.id = e.from_id {edge_type_filter}
    JOIN nodes n ON e.to_id = n.id
    WHERE t.depth < {max_depth}
      AND n.id != ALL(t.path)  -- Cycle detection
)
SELECT t.*, n.*
FROM traverse t
JOIN nodes n ON t.id = n.id
WHERE t.depth >= {min_depth}
{end_label_filter}
{user_where}
ORDER BY t.depth, t.amplification DESC
{limit}
"#,
            start_where = start_where,
            edge_type_filter = edge_type_filter,
            max_depth = edge.max_hops,
            min_depth = edge.min_hops,
            end_label_filter = if let Some(ref end) = end_node {
                if !end.labels.is_empty() {
                    format!("  AND n.label = '{}'", end.labels[0])
                } else {
                    String::new()
                }
            } else {
                String::new()
            },
            user_where = if let Some(ref w) = query.where_clause {
                format!("  AND ({})", Self::condition_to_sql(&w.condition))
            } else {
                String::new()
            },
            limit = query.limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default(),
        );
        
        Ok(sql)
    }
    
    fn extract_path_pattern(pattern: &Pattern) -> Result<(NodePattern, EdgePattern, Option<NodePattern>)> {
        let start = match pattern.elements.first() {
            Some(PatternElement::Node(n)) => n.clone(),
            _ => return Err(Error::Query("Pattern must start with a node".into())),
        };
        
        let edge = match pattern.elements.get(1) {
            Some(PatternElement::Edge(e)) => e.clone(),
            _ => return Err(Error::Query("Pattern must have an edge".into())),
        };
        
        let end = match pattern.elements.get(2) {
            Some(PatternElement::Node(n)) => Some(n.clone()),
            _ => None,
        };
        
        Ok((start, edge, end))
    }
    
    fn transpile_create(query: &CypherQuery) -> Result<String> {
        let create_clause = query.create_clause.as_ref()
            .ok_or_else(|| Error::Query("Missing CREATE clause".into()))?;
        
        let mut sql = String::new();
        
        for pattern in &create_clause.patterns {
            for element in &pattern.elements {
                match element {
                    PatternElement::Node(node) => {
                        let id = node.properties.get("id")
                            .map(|v| Self::value_to_sql(v))
                            .unwrap_or_else(|| format!("'{}'", uuid::Uuid::new_v4()));
                        
                        let label = node.labels.first()
                            .map(|l| format!("'{}'", l))
                            .unwrap_or_else(|| "'Node'".to_string());
                        
                        let props = serde_json::to_string(&node.properties)
                            .unwrap_or_else(|_| "{}".to_string());
                        
                        sql.push_str(&format!(
                            "INSERT INTO nodes (id, label, properties) VALUES ({}, {}, '{}');\n",
                            id, label, props
                        ));
                    }
                    PatternElement::Edge(edge) => {
                        // Edge creation requires knowing the from/to node IDs
                        // This is simplified - real implementation needs alias resolution
                    }
                }
            }
        }
        
        Ok(sql)
    }
    
    fn value_to_sql(value: &Value) -> String {
        match value {
            Value::String(s) => format!("'{}'", s.replace('\'', "''")),
            Value::Integer(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Boolean(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
            Value::Null => "NULL".to_string(),
            Value::List(items) => {
                let vals = items.iter().map(Self::value_to_sql).collect::<Vec<_>>().join(", ");
                format!("ARRAY[{}]", vals)
            }
        }
    }
    
    fn expr_to_sql(expr: &Expr) -> String {
        match expr {
            Expr::Property { alias, property } => format!("{}.{}", alias, property),
            Expr::Literal(v) => Self::value_to_sql(v),
            Expr::Variable(v) => format!("{}.*", v),
            Expr::Function { name, args } => {
                let args_sql = args.iter().map(Self::expr_to_sql).collect::<Vec<_>>().join(", ");
                format!("{}({})", name, args_sql)
            }
        }
    }
    
    fn condition_to_sql(cond: &Condition) -> String {
        match cond {
            Condition::Comparison { left, op, right } => {
                let op_str = match op {
                    ComparisonOp::Eq => "=",
                    ComparisonOp::Ne => "<>",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => ">=",
                    ComparisonOp::Contains => "LIKE",
                    ComparisonOp::StartsWith => "LIKE",
                    ComparisonOp::EndsWith => "LIKE",
                };
                
                let right_sql = match op {
                    ComparisonOp::Contains => {
                        if let Expr::Literal(Value::String(s)) = right {
                            format!("'%{}%'", s)
                        } else {
                            Self::expr_to_sql(right)
                        }
                    }
                    ComparisonOp::StartsWith => {
                        if let Expr::Literal(Value::String(s)) = right {
                            format!("'{}%'", s)
                        } else {
                            Self::expr_to_sql(right)
                        }
                    }
                    ComparisonOp::EndsWith => {
                        if let Expr::Literal(Value::String(s)) = right {
                            format!("'%{}'", s)
                        } else {
                            Self::expr_to_sql(right)
                        }
                    }
                    _ => Self::expr_to_sql(right),
                };
                
                format!("{} {} {}", Self::expr_to_sql(left), op_str, right_sql)
            }
            Condition::And(left, right) => {
                format!("({}) AND ({})", Self::condition_to_sql(left), Self::condition_to_sql(right))
            }
            Condition::Or(left, right) => {
                format!("({}) OR ({})", Self::condition_to_sql(left), Self::condition_to_sql(right))
            }
            Condition::Not(inner) => {
                format!("NOT ({})", Self::condition_to_sql(inner))
            }
            Condition::IsNull(expr) => {
                format!("{} IS NULL", Self::expr_to_sql(expr))
            }
            Condition::IsNotNull(expr) => {
                format!("{} IS NOT NULL", Self::expr_to_sql(expr))
            }
            Condition::In(expr, values) => {
                let vals = values.iter().map(Self::value_to_sql).collect::<Vec<_>>().join(", ");
                format!("{} IN ({})", Self::expr_to_sql(expr), vals)
            }
        }
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Parse and transpile Cypher to SQL
pub fn cypher_to_sql(cypher: &str) -> Result<String> {
    let query = CypherParser::parse(cypher)?;
    CypherTranspiler::transpile(&query)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_match() {
        let cypher = "MATCH (a:Thought)-[:CAUSES]->(b:Thought) RETURN b";
        let sql = cypher_to_sql(cypher).unwrap();
        assert!(sql.contains("SELECT"));
        assert!(sql.contains("JOIN edges"));
        assert!(sql.contains("type IN ('CAUSES')"));
    }
    
    #[test]
    fn test_variable_length() {
        let cypher = "MATCH (a)-[:CAUSES*1..5]->(b) RETURN b";
        let sql = cypher_to_sql(cypher).unwrap();
        assert!(sql.contains("WITH RECURSIVE"));
        assert!(sql.contains("depth < 5"));
    }
    
    #[test]
    fn test_where_clause() {
        let cypher = "MATCH (a:Thought) WHERE a.qidx > 100 RETURN a";
        let sql = cypher_to_sql(cypher).unwrap();
        assert!(sql.contains("a.qidx > 100"));
    }
    
    #[test]
    fn test_multi_type_edge() {
        let cypher = "MATCH (a)-[:CAUSES|ENABLES]->(b) RETURN b";
        let sql = cypher_to_sql(cypher).unwrap();
        assert!(sql.contains("'CAUSES'"));
        assert!(sql.contains("'ENABLES'"));
    }
}
